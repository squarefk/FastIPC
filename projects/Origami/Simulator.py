from reader import *
from common.math.math_tools import *
from common.utils.timer import *
from common.utils.logger import *

import sys, os, time, math
import taichi as ti
import taichi_three as t3
import numpy as np
import meshio
import pickle
import scipy.sparse
import scipy.sparse.linalg
from sksparse.cholmod import *
from dihedral_angle import *
from simplex_volume import *

##############################################################################
testcase = int(sys.argv[1])
settings = read()
mesh_particles = settings['mesh_particles']
mesh_elements = settings['mesh_elements']
mesh_edges = settings['mesh_edges']
dim = 3
codim = 2
gravity = settings['gravity']
thickness = 0.0003
E = 1e9
nu = 0.3
base_bending_weight = 10 # E * thickness ** 3 / (24 * (1 - nu * nu))
quasi_static = True

##############################################################################
directory = 'output/' + '_'.join(sys.argv[:2]) + '/'
os.makedirs(directory + 'images/', exist_ok=True)
os.makedirs(directory + 'caches/', exist_ok=True)
os.makedirs(directory + 'objs/', exist_ok=True)
print('output directory:', directory)
# sys.stdout = open(directory + 'log.txt', 'w')
# sys.stderr = open(directory + 'err.txt', 'w')

##############################################################################
real = ti.f64
ti.init(arch=ti.cpu, default_fp=real)#, cpu_max_num_threads=1)

scalar = lambda: ti.field(real)
vec = lambda: ti.Vector.field(dim, real)
mat = lambda: ti.Matrix.field(dim, dim, real)
mat2 = lambda: ti.Matrix.field(codim, codim, real)

dt = 0.04
sub_steps = 1
n_particles = len(mesh_particles)
n_elements = len(mesh_elements)
n_edges = len(mesh_edges)
n_boundary_triangles = n_elements

x0, x, xPrev, xTilde, xn, v, m, vol0, rho = vec(), vec(), vec(), vec(), vec(), vec(), scalar(), scalar(), scalar()
zero = vec()
la, mu = scalar(), scalar()
B = mat2()
vertices = ti.field(ti.i32)
edges = ti.field(ti.i32)
rest_angle = ti.field(real)
rest_e = ti.field(real)
rest_h = ti.field(real)
weight = ti.field(real)
ti.root.dense(ti.i, n_particles).place(x, x0, xPrev, xTilde, xn, v, m)
ti.root.dense(ti.i, n_particles).place(zero)
ti.root.dense(ti.i, n_elements).place(la, mu, vol0, rho)
ti.root.dense(ti.i, n_elements).place(B)
ti.root.dense(ti.ij, (n_elements, dim)).place(vertices)
ti.root.dense(ti.ij, (n_edges, dim + 2)).place(edges)
ti.root.dense(ti.i, n_edges).place(rest_angle, rest_e, rest_h, weight)

MAX_LINEAR = n_particles * 3 + 144 * n_edges + 81 * n_elements + 10
data_rhs = ti.field(real, shape=n_particles * dim)
data_rhs_fd = ti.field(real, shape=n_particles * dim)
data_row = ti.field(ti.i32, shape=MAX_LINEAR)
data_col = ti.field(ti.i32, shape=MAX_LINEAR)
data_val = ti.field(real, shape=MAX_LINEAR)
cnt = ti.field(ti.i32, shape=())

hessianXx_row = ti.field(ti.i32, shape=MAX_LINEAR)
hessianXx_col = ti.field(ti.i32, shape=MAX_LINEAR)
hessianXx_val = ti.field(real, shape=MAX_LINEAR)
cntXx = ti.field(ti.i32, shape=())

data_sol = ti.field(real, shape=n_particles * dim)

dfx = ti.field(ti.i32, shape=n_particles * dim)

v_ = ti.Vector.field(dim, real, shape=4)

@ti.func
def compute_density(i):
    return 800.


@ti.func
def compute_lame_parameters(i):
    # E = 0.
    # if testcase == 1002:
    #     E = 1e9
    # else:
    #     E = 1e9
    # E = 3e9
    # E = 1e5
    # nu = 0.3
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


@ti.kernel
def compute_filter_3D_inversion_free() -> real:
    alpha = 1.0
    for i in range(n_elements):
        a, b, c, d = vertices[i, 0], vertices[i, 1], vertices[i, 2], vertices[i, 3]
        da = ti.Vector([data_sol[a * dim + 0], data_sol[a * dim + 1], data_sol[a * dim + 2]])
        db = ti.Vector([data_sol[b * dim + 0], data_sol[b * dim + 1], data_sol[b * dim + 2]])
        dc = ti.Vector([data_sol[c * dim + 0], data_sol[c * dim + 1], data_sol[c * dim + 2]])
        dd = ti.Vector([data_sol[d * dim + 0], data_sol[d * dim + 1], data_sol[d * dim + 2]])
        alpha = min(alpha, get_smallest_positive_real_cubic_root(x[a], x[b], x[c], x[d], da, db, dc, dd, 0.2))
    return alpha


@ti.func
def compute_T(i):
    if ti.static(codim == 2):
        ab = x[vertices[i, 1]] - x[vertices[i, 0]]
        ac = x[vertices[i, 2]] - x[vertices[i, 0]]
        T = ti.Matrix.cols([ab, ac])
        return T.transpose() @ T

@ti.kernel
def init_material():
    for i in range(n_elements):
        rho[i] = compute_density(i)
        la[i], mu[i] = compute_lame_parameters(i)

    for i in range(n_edges):
        if edges[i, 4] == 1:
            weight[i] = 100 * base_bending_weight
        elif edges[i, 4] == -1:
            weight[i] = 10 * base_bending_weight
        else:
            weight[i] = base_bending_weight

@ti.kernel
def reset():
    for i in range(n_elements):
        ab = x0[vertices[i, 1]] - x0[vertices[i, 0]]
        ac = x0[vertices[i, 2]] - x0[vertices[i, 0]]
        T = ti.Matrix.cols([ab, ac])
        B[i] = (T.transpose() @ T).inverse()
        vol0[i] = thickness * (ab.cross(ac)).norm() / 2
        mass = vol0[i] * rho[i] / (codim + 1)
        if mass < 0.0:
            print("FATAL ERROR : mesh inverted")
        for d in ti.static(range(dim + 1)):
            m[vertices[i, d]] += mass
        
    for i in range(n_edges):
        rest_e[i] = (x0[edges[i, 0]] - x0[edges[i, 1]]).norm()
        if edges[i, 3] < 0:
            rest_h[i] = 1 
            continue
        rest_angle[i] = 0.0
        X0 = x0[edges[i, 2]]
        X1 = x0[edges[i, 0]]
        X2 = x0[edges[i, 1]]
        X3 = x0[edges[i, 3]]
        n1 = (X1 - X0).cross(X2 - X0)
        n2 = (X2 - X3).cross(X1 - X3)
        rest_h[i] = (n1.norm() + n2.norm()) / (rest_e[i] * 6)


@ti.kernel
def compute_xn_and_xTilde():
    for i in range(n_particles):
        xn[i] = x[i]
        xTilde[i] = x[i] + dt * v[i]
        xTilde(1)[i] += dt * dt * gravity


def move_nodes(current_time):
    dirichlet_fixed, dirichlet_value = settings['dirichlet'](current_time)
    for i in range(n_particles):
        if dirichlet_fixed[i]:
            for d in range(dim):
                xTilde(d)[i] = dirichlet_value[i, d]
                x(d)[i] = dirichlet_value[i, d]
    rest_angle.from_numpy(settings['rest_angle'](current_time))

@ti.kernel
def check_edge_error() -> real:
    edge_error = 0
    for e in range(n_edges):
        l = (x[edges[e, 0]] - x[edges[e, 1]]).norm()
        edge_error += ti.abs(l - rest_e[e]) / rest_e[e]
    return edge_error

@ti.kernel
def compute_energy() -> real:
    total_energy = 0.0
    # inertia
    for i in range(n_particles):
        total_energy += 0.5 * m[i] * (x[i] - xTilde[i]).norm_sqr()
    
    # membrane
    for e in range(n_elements):
        F = compute_T(e) @ B[e]
        lnJ = 0.5 * ti.log(F.determinant())
        mem = 0.5 * mu[e] * (F.trace() - 2 - 2 * lnJ) + 0.5 * la[e] * lnJ * lnJ
        total_energy += mem * dt * dt * vol0[e]
    
    # bending
    for e in range(n_edges):
        if edges[e, 3] < 0: continue
        x0 = x[edges[e, 2]]
        x1 = x[edges[e, 0]]
        x2 = x[edges[e, 1]]
        x3 = x[edges[e, 3]]
        theta = dihedral_angle(x0, x1, x2, x3, edges[e, 4])
        ben = (theta - rest_angle[e]) * (theta - rest_angle[e]) * rest_e[e] / rest_h[e]
        total_energy += weight[e] * dt * dt * ben
    
    return total_energy

@ti.kernel
def compute_gradient():
    # negtive gradient
    # inertia
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            data_rhs[i * dim + d] -= m[i] * (x(d)[i] - xTilde(d)[i])
    
    # membrane
    for e in range(n_elements):
        x1, x2, x3 = x[vertices[e, 0]], x[vertices[e, 1]], x[vertices[e, 2]]
        A = compute_T(e)
        IA = A.inverse()
        IB = B[e]
        lnJ = 0.5 * ti.log(A.determinant() * IB.determinant())
        de_div_dA = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for i in ti.static(range(2)):
            for j in ti.static(range(2)):
                de_div_dA[j * codim + i] = dt * dt * vol0[e] * ((0.5 * mu[e] * IB[i,j] + 0.5 * (-mu[e] + la[e] * lnJ) * IA[i,j]))
        Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dA_div_dx = ti.Matrix.rows([Z, Z, Z, Z])
        for i in ti.static(range(3)):
            dA_div_dx[0, 3 + i] += 2.0 * (x2[i] - x1[i])
            dA_div_dx[0, 0 + i] -= 2.0 * (x2[i] - x1[i])
            dA_div_dx[1, 6 + i] += (x2[i] - x1[i])
            dA_div_dx[1, 3 + i] += (x3[i] - x1[i])
            dA_div_dx[1, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
            dA_div_dx[2, 6 + i] += (x2[i] - x1[i])
            dA_div_dx[2, 3 + i] += (x3[i] - x1[i])
            dA_div_dx[2, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
            dA_div_dx[3, 6 + i] += 2.0 * (x3[i] - x1[i])
            dA_div_dx[3, 0 + i] -= 2.0 * (x3[i] - x1[i])
        
        grad = dA_div_dx.transpose() @ de_div_dA
        indMap = ti.Vector([vertices[e, 0] * 3, vertices[e, 0] * 3 + 1, vertices[e, 0] * 3 + 2,
                            vertices[e, 1] * 3, vertices[e, 1] * 3 + 1, vertices[e, 1] * 3 + 2,
                            vertices[e, 2] * 3, vertices[e, 2] * 3 + 1, vertices[e, 2] * 3 + 2])
        
        for i in ti.static(range(9)):
            data_rhs[indMap[i]] -= grad[i]
    
    # bending
    for e in range(n_edges):
        if edges[e, 3] < 0: continue
        x0 = x[edges[e, 2]]
        x1 = x[edges[e, 0]]
        x2 = x[edges[e, 1]]
        x3 = x[edges[e, 3]]
        theta = dihedral_angle(x0, x1, x2, x3, edges[e, 4])
        grad = dihedral_angle_gradient(x0, x1, x2, x3)
        grad *= weight[e] * dt * dt * 2 * (theta - rest_angle[e]) * rest_e[e] / rest_h[e]
        for d in ti.static(range(3)):
            data_rhs[3 * edges[e, 2] + d] -= grad[0 * 3 + d]
            data_rhs[3 * edges[e, 0] + d] -= grad[1 * 3 + d]
            data_rhs[3 * edges[e, 1] + d] -= grad[2 * 3 + d]
            data_rhs[3 * edges[e, 3] + d] -= grad[3 * 3 + d]


def check_gradient():
    x.from_numpy(x.to_numpy() * 10)
    xTilde.from_numpy(x.to_numpy())
    la.from_numpy(np.ones((n_elements, )))
    mu.from_numpy(np.ones((n_elements, )))
    m.from_numpy(np.zeros((n_particles, )))
    weight.from_numpy(np.ones(n_edges, ))
    global dt
    dt = 1
    # dt = 100
    # rest_angle.from_numpy(np.pi * 0.3 * np.ones((n_edges,)))
    # rest_angle.from_numpy(np.zeros((n_edges,)))
    n = n_particles * dim

    eps = 1e-6

    x_bk = x.to_numpy()
    for i in range(100):
        delta_x = eps * (np.random.rand(n_particles, 3) * 2 - 1)
        x.from_numpy(x_bk + delta_x)
        E1 = compute_energy()
        data_rhs.fill(0)
        compute_gradient()
        g1 = - data_rhs.to_numpy()
        x.from_numpy(x_bk - delta_x)
        E0 = compute_energy()
        data_rhs.fill(0)
        compute_gradient()
        g0 = - data_rhs.to_numpy()
        print((E1 - E0 - np.dot(g1 + g0, delta_x.flatten())) / eps, (E1 - E0 - 2 * np.dot(g1 + g0, delta_x.flatten())) / eps)
        input()

    # data_rhs.fill(0)
    # compute_gradient()
    # x_bk = x.to_numpy().flatten()
    # for n in range(n_particles * dim):
    #     x_copy = x_bk.copy()
    #     x_copy[n] -= eps
    #     x.from_numpy(np.reshape(x_copy, (n_particles, 3)))
    #     e0 = compute_energy()

    #     x_copy = x_bk.copy()
    #     x_copy[n] += eps
    #     x.from_numpy(np.reshape(x_copy, (n_particles, 3)))
    #     e1 = compute_energy()
    #     print((e1 - e0) / (2 * eps), -data_rhs[n])
    #     input()


@ti.kernel
def compute_hessian(pd: ti.int32):
    cnt[None] = 0
    # inertia
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            c = cnt[None] + i * dim + d
            data_row[c] = i * dim + d
            data_col[c] = i * dim + d
            data_val[c] = m[i]
    cnt[None] += n_particles * dim

    # membrane
    Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ahess = [ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z]), 
             ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z]), 
             ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z]), 
             ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z])]
    for d in ti.static(range(dim)):
        ahess[0][0 + d, 0 + d] += 2.0
        ahess[0][3 + d, 3 + d] += 2.0
        ahess[0][0 + d, 3 + d] -= 2.0
        ahess[0][3 + d, 0 + d] -= 2.0

        ahess[1][3 + d, 6 + d] += 1.0
        ahess[1][6 + d, 3 + d] += 1.0
        ahess[1][0 + d, 3 + d] -= 1.0
        ahess[1][0 + d, 6 + d] -= 1.0
        ahess[1][3 + d, 0 + d] -= 1.0
        ahess[1][6 + d, 0 + d] -= 1.0
        ahess[1][0 + d, 0 + d] += 2.0

        ahess[2][3 + d, 6 + d] += 1.0
        ahess[2][6 + d, 3 + d] += 1.0
        ahess[2][0 + d, 3 + d] -= 1.0
        ahess[2][0 + d, 6 + d] -= 1.0
        ahess[2][3 + d, 0 + d] -= 1.0
        ahess[2][6 + d, 0 + d] -= 1.0
        ahess[2][0 + d, 0 + d] += 2.0

        ahess[3][0 + d, 0 + d] += 2.0
        ahess[3][6 + d, 6 + d] += 2.0
        ahess[3][0 + d, 6 + d] -= 2.0
        ahess[3][6 + d, 0 + d] -= 2.0

    for e in range(n_elements):
        x1, x2, x3 = x[vertices[e, 0]], x[vertices[e, 1]], x[vertices[e, 2]]
        IB = B[e]
        A = compute_T(e)
        Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dA_div_dx = ti.Matrix.rows([Z, Z, Z, Z])
        for i in ti.static(range(3)):
            dA_div_dx[0, 3 + i] += 2.0 * (x2[i] - x1[i])
            dA_div_dx[0, 0 + i] -= 2.0 * (x2[i] - x1[i])
            dA_div_dx[1, 6 + i] += (x2[i] - x1[i])
            dA_div_dx[1, 3 + i] += (x3[i] - x1[i])
            dA_div_dx[1, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
            dA_div_dx[2, 6 + i] += (x2[i] - x1[i])
            dA_div_dx[2, 3 + i] += (x3[i] - x1[i])
            dA_div_dx[2, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
            dA_div_dx[3, 6 + i] += 2.0 * (x3[i] - x1[i])
            dA_div_dx[3, 0 + i] -= 2.0 * (x3[i] - x1[i])
        
        IA = A.inverse()
        ainvda = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for endI in ti.static(range(3)):
            for dimI in ti.static(range(3)):
                ainvda[endI * dim + dimI] = \
                    dA_div_dx[0, endI * dim + dimI] * IA[0, 0] + \
                    dA_div_dx[1, endI * dim + dimI] * IA[1, 0] + \
                    dA_div_dx[2, endI * dim + dimI] * IA[0, 1] + \
                    dA_div_dx[3, endI * dim + dimI] * IA[1, 1]
        deta = A.determinant()
        lnJ = 0.5 * ti.log(deta * IB.determinant())
        term1 = (-mu[e] + la[e] * lnJ) * 0.5
        hessian = (-term1 + 0.25 * la[e]) * (ainvda @ ainvda.transpose())
        aderivadj = ti.Matrix.rows([Z, Z, Z, Z])
        for d in ti.static(range(9)):
            aderivadj[0, d] = dA_div_dx[3, d]
            aderivadj[1, d] = - dA_div_dx[1, d]
            aderivadj[2, d] = - dA_div_dx[2, d]
            aderivadj[3, d] = dA_div_dx[0, d]
        hessian += term1 / deta * aderivadj.transpose() @ dA_div_dx
        for i in ti.static(range(2)):
            for j in ti.static(range(2)):
                hessian += (term1 * IA[i, j] + 0.5 * mu[e] * IB[i, j]) * ahess[i + j * 2]
        hessian *= (dt * dt * vol0[e])
    
        if pd == 1:
            hessian = project_pd(hessian)
        
        indMap = ti.Vector([vertices[e, 0] * 3, vertices[e, 0] * 3 + 1, vertices[e, 0] * 3 + 2,
                            vertices[e, 1] * 3, vertices[e, 1] * 3 + 1, vertices[e, 1] * 3 + 2,
                            vertices[e, 2] * 3, vertices[e, 2] * 3 + 1, vertices[e, 2] * 3 + 2])
        for i in ti.static(range(9)):
            for j in ti.static(range(9)):
                if dfx[indMap[i]] or dfx[indMap[j]]:
                    hessian[i, j] = 0
                c = cnt[None] + e * 81 + i * 9 + j
                data_row[c], data_col[c], data_val[c] = indMap[i], indMap[j], hessian[i, j]
    cnt[None] += 81 * n_elements

    # bending
    for e in range(n_edges):
        if edges[e, 3] < 0: continue
        x0 = x[edges[e, 2]]
        x1 = x[edges[e, 0]]
        x2 = x[edges[e, 1]]
        x3 = x[edges[e, 3]]
        theta = dihedral_angle(x0, x1, x2, x3, edges[e, 4])
        grad = dihedral_angle_gradient(x0, x1, x2, x3)
        H = dihedral_angle_hessian(x0, x1, x2, x3)
        H *= dt * dt * weight[e] * 2.0 * (theta - rest_angle[e]) * rest_e[e] / rest_h[e]
        H += (dt * dt * weight[e] * 2.0 * rest_e[e] / rest_h[e]) * grad @ grad.transpose()
        
        if pd: 
            H = project_pd(H)
        indMap = ti.Vector([edges[e, 2] * 3, edges[e, 2] * 3 + 1, edges[e, 2] * 3 + 2,
                            edges[e, 0] * 3, edges[e, 0] * 3 + 1, edges[e, 0] * 3 + 2,
                            edges[e, 1] * 3, edges[e, 1] * 3 + 1, edges[e, 1] * 3 + 2,
                            edges[e, 3] * 3, edges[e, 3] * 3 + 1, edges[e, 3] * 3 + 2])
        for i in ti.static(range(12)):
            for j in ti.static(range(12)):
                if dfx[indMap[i]] or dfx[indMap[j]]:
                    H[i, j] = 0
                c = cnt[None] + e * 144 + i * 12 + j
                data_row[c], data_col[c], data_val[c] = indMap[i], indMap[j], H[i, j]
    cnt[None] += 144 * n_edges

@ti.kernel
def compute_hessian_Xx():
    cntXx[None] = 0
    
    # membrane
    for e in range(n_elements):
        x1, x2, x3 = x[vertices[e, 0]], x[vertices[e, 1]], x[vertices[e, 2]]
        X1, X2, X3 = x0[vertices[e, 0]], x0[vertices[e, 1]], x0[vertices[e, 2]]
        IB = B[e]
        A = compute_T(e)
        IA = A.inverse()
        lnJ = 0.5 * ti.log(A.determinant() * IB.determinant())
        dv_div_dX = thickness * simplex_volume_gradient(X1, X2, X3)
        de_div_dA = ti.Vector([0.,0.,0.,0.])
        for i in ti.static(range(2)):
            for j in ti.static(range(2)):
                de_div_dA[j * codim + i] = ((0.5 * mu[e] * IB[i,j] + 0.5 * (-mu[e] + la[e] * lnJ) * IA[i,j]))
        Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dA_div_dx = ti.Matrix.rows([Z, Z, Z, Z])
        dB_div_dX = ti.Matrix.rows([Z, Z, Z, Z])
        for i in ti.static(range(3)):
            dA_div_dx[0, 3 + i] += 2.0 * (x2[i] - x1[i])
            dA_div_dx[0, 0 + i] -= 2.0 * (x2[i] - x1[i])
            dA_div_dx[1, 6 + i] += (x2[i] - x1[i])
            dA_div_dx[1, 3 + i] += (x3[i] - x1[i])
            dA_div_dx[1, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
            dA_div_dx[2, 6 + i] += (x2[i] - x1[i])
            dA_div_dx[2, 3 + i] += (x3[i] - x1[i])
            dA_div_dx[2, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
            dA_div_dx[3, 6 + i] += 2.0 * (x3[i] - x1[i])
            dA_div_dx[3, 0 + i] -= 2.0 * (x3[i] - x1[i])

            dB_div_dX[0, 3 + i] += 2.0 * (X2[i] - X1[i])
            dB_div_dX[0, 0 + i] -= 2.0 * (X2[i] - X1[i])
            dB_div_dX[1, 6 + i] += (X2[i] - X1[i])
            dB_div_dX[1, 3 + i] += (X3[i] - X1[i])
            dB_div_dX[1, 0 + i] += - (X2[i] - X1[i]) - (X3[i] - X1[i])
            dB_div_dX[2, 6 + i] += (X2[i] - X1[i])
            dB_div_dX[2, 3 + i] += (X3[i] - X1[i])
            dB_div_dX[2, 0 + i] += - (X2[i] - X1[i]) - (X3[i] - X1[i])
            dB_div_dX[3, 6 + i] += 2.0 * (X3[i] - X1[i])
            dB_div_dX[3, 0 + i] -= 2.0 * (X3[i] - X1[i])
            
        de_div_dx = dA_div_dx.transpose() @ de_div_dA

        # first term
        hessian = dt * dt * dv_div_dX @ de_div_dx.transpose()

        # second term
        Z4 = ti.Vector([0.,0.,0.,0.])
        dbinv_div_db = ti.Matrix.rows([Z4, Z4, Z4, Z4])
        for m in ti.static(range(2)):
            for n in ti.static(range(2)):
                for i in ti.static(range(2)):
                    for j in ti.static(range(2)): 
                        dbinv_div_db[n * codim + m, j * codim + i] = - IB[m, i] * IB[j, n]
        
        d2e_divA_divB = 0.5 * mu[e] * dbinv_div_db
        for m in ti.static(range(2)):
            for n in ti.static(range(2)):
                for i in ti.static(range(2)):
                    for j in ti.static(range(2)):
                        d2e_divA_divB[n * codim + m, j * codim + i] -= 0.25 * la[e] * IA[m, n] * IB[j, i]
        hessian += dt * dt * vol0[e] * dB_div_dX.transpose() @ d2e_divA_divB.transpose() @ dA_div_dx
        
        indMap = ti.Vector([vertices[e, 0] * 3, vertices[e, 0] * 3 + 1, vertices[e, 0] * 3 + 2,
                            vertices[e, 1] * 3, vertices[e, 1] * 3 + 1, vertices[e, 1] * 3 + 2,
                            vertices[e, 2] * 3, vertices[e, 2] * 3 + 1, vertices[e, 2] * 3 + 2])
        for i in ti.static(range(9)):
            for j in ti.static(range(9)):
                if dfx[indMap[j]]:
                    hessian[i, j] = 0
                c = cntXx[None] + e * 81 + i * 9 + j
                hessianXx_row[c], hessianXx_col[c], hessianXx_val[c] = indMap[i], indMap[j], hessian[i, j]
    cntXx[None] += 81 * n_elements

    # bending
    for e in range(n_edges):
        if edges[e, 3] < 0: continue
        x1 = x[edges[e, 2]]
        x2 = x[edges[e, 0]]
        x3 = x[edges[e, 1]]
        x4 = x[edges[e, 3]]

        X1 = x0[edges[e, 2]]
        X2 = x0[edges[e, 0]]
        X3 = x0[edges[e, 1]]
        X4 = x0[edges[e, 3]]

        theta = dihedral_angle(x1, x2, x3, x4, edges[e, 4])
        grad = dihedral_angle_gradient(x1, x2, x3, x4)

        dA1_div_dX = simplex_volume_gradient(X1, X2, X3)
        dA2_div_dX = simplex_volume_gradient(X2, X3, X4)

        dAsum_div_dX = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for d in ti.static(range(3)):
            dAsum_div_dX[dim * 0 + d] = dA1_div_dX[dim * 0 + d] 
            dAsum_div_dX[dim * 1 + d] = dA1_div_dX[dim * 1 + d] + dA2_div_dX[dim * 0 + d]
            dAsum_div_dX[dim * 2 + d] = dA1_div_dX[dim * 2 + d] + dA2_div_dX[dim * 1 + d]
            dAsum_div_dX[dim * 3 + d] =                           dA2_div_dX[dim * 2 + d]
        
        n = (X2 - X3).normalized()
        dl_div_dX = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for d in ti.static(range(3)):
            dl_div_dX[dim * 1 + d] = n[d]
            dl_div_dX[dim * 2 + d] = - n[d]

        H = weight[e] * dt * dt * 2 * (theta - rest_angle[e]) / rest_h[e] *  (2 * dl_div_dX - dAsum_div_dX / (3 * rest_h[e])) @ grad.transpose()
        
        indMap = ti.Vector([edges[e, 2] * 3, edges[e, 2] * 3 + 1, edges[e, 2] * 3 + 2,
                            edges[e, 0] * 3, edges[e, 0] * 3 + 1, edges[e, 0] * 3 + 2,
                            edges[e, 1] * 3, edges[e, 1] * 3 + 1, edges[e, 1] * 3 + 2,
                            edges[e, 3] * 3, edges[e, 3] * 3 + 1, edges[e, 3] * 3 + 2])
        for i in ti.static(range(12)):
            for j in ti.static(range(12)):
                if dfx[indMap[j]]:
                    H[i, j] = 0
                c = cntXx[None] + e * 144 + i * 12 + j
                hessianXx_row[c], hessianXx_col[c], hessianXx_val[c] = indMap[i], indMap[j], H[i, j]
    cntXx[None] += 144 * n_edges


def check_hessian():
    # input()
    # x.from_numpy(x.to_numpy() * 2)
    xTilde.from_numpy(x.to_numpy())
    la.from_numpy(np.ones((n_elements, )))
    mu.from_numpy(np.ones((n_elements, )))
    m.from_numpy(np.zeros((n_particles, )))
    global dt
    dt = 1
    # vol0.from_numpy(np.ones((n_elements, )))
    # rest_angle.from_numpy(np.zeros((n_edges,)))
    n = n_particles * dim

    eps = 1e-4
    x_bk = x.to_numpy()
    # for i in range(100):
    #     delta_x = eps * (np.random.rand(n_particles, 3) * 2 - 1)
    #     x.from_numpy(x_bk + delta_x)
    #     data_rhs.fill(0)
    #     compute_gradient()
    #     g1 = - data_rhs.to_numpy()
    #     data_row.fill(0)
    #     data_col.fill(0)
    #     data_val.fill(0)
    #     compute_hessian(0)
    #     row, col, val = data_row.to_numpy()[:cnt[None]], data_col.to_numpy()[:cnt[None]], data_val.to_numpy()[:cnt[None]]
    #     A1 = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n))

    #     x.from_numpy(x_bk - delta_x)
    #     data_rhs.fill(0)
    #     compute_gradient()
    #     g0 = - data_rhs.to_numpy()
    #     data_row.fill(0)
    #     data_col.fill(0)
    #     data_val.fill(0)
    #     compute_hessian(0)
    #     row, col, val = data_row.to_numpy()[:cnt[None]], data_col.to_numpy()[:cnt[None]], data_val.to_numpy()[:cnt[None]]
    #     A0 = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n))
    #     print(np.linalg.norm(g1 - g0 - (A1 + A0).dot(delta_x.flatten())) / eps, np.linalg.norm(g1 - g0 - 2 * (A1 + A0).dot(delta_x.flatten())) / eps)
    #     import pdb
    #     pdb.set_trace()

    data_row.fill(0)
    data_col.fill(0)
    data_val.fill(0)
    compute_hessian(0)
    row, col, val = data_row.to_numpy()[:cnt[None]], data_col.to_numpy()[:cnt[None]], data_val.to_numpy()[:cnt[None]]
    A = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n))
    rows,cols = A.nonzero()
    for row,col in zip(rows,cols):
        x_copy = x_bk.copy()
        x_copy[col // 3, col % 3] -= eps
        x.from_numpy(x_copy)
        data_rhs.fill(0)
        compute_gradient()
        g0 = - data_rhs[row]

        x_copy = x_bk.copy()
        x_copy[col // 3, col % 3] += eps
        x.from_numpy(x_copy)
        data_rhs.fill(0)
        compute_gradient()
        g1 = - data_rhs[row]
        print((g1 - g0) / (2 * eps), A[row, col])
        input()
      

def check_hessian_Xx():
    x0.from_numpy(10 * x0.to_numpy())
    reset()
    la.from_numpy(np.random.random((n_elements, )))
    mu.from_numpy(np.random.random((n_elements, )))
    m.from_numpy(np.zeros((n_particles, )))
    weight.from_numpy(np.random.random(n_edges, ))
    global dt
    dt = 0.152
    rest_angle.from_numpy(np.zeros((n_edges,)))
    n = n_particles * dim
    eps = 1e-4
    x_bk = x0.to_numpy()
    for i in range(100):
        delta_x = eps * (np.random.rand(n_particles, 3) * 2 - 1)
        x0.from_numpy(x_bk + delta_x)
        reset()
        data_rhs.fill(0)
        compute_gradient()
        g1 = - data_rhs.to_numpy()
        hessianXx_row.fill(0)
        hessianXx_col.fill(0)
        hessianXx_val.fill(0)
        compute_hessian_Xx()
        row, col, val = hessianXx_row.to_numpy()[:cntXx[None]], hessianXx_col.to_numpy()[:cntXx[None]], hessianXx_val.to_numpy()[:cntXx[None]]
        A1 = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n)).transpose()

        x0.from_numpy(x_bk - delta_x)
        reset()
        data_rhs.fill(0)
        compute_gradient()
        g0 = - data_rhs.to_numpy()
        hessianXx_row.fill(0)
        hessianXx_col.fill(0)
        hessianXx_val.fill(0)
        compute_hessian_Xx()
        row, col, val = hessianXx_row.to_numpy()[:cntXx[None]], hessianXx_col.to_numpy()[:cntXx[None]], hessianXx_val.to_numpy()[:cntXx[None]]
        A0 = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n)).transpose()
        print(np.linalg.norm(g1 - g0 - (A1 + A0).dot(delta_x.flatten())) / eps, np.linalg.norm(g1 - g0 - 2 * (A1 + A0).dot(delta_x.flatten())) / eps)
        input()

    # hessianXx_row.fill(0)
    # hessianXx_col.fill(0)
    # hessianXx_val.fill(0)
    # compute_hessian_Xx()
    # row, col, val = hessianXx_row.to_numpy()[:cntXx[None]], hessianXx_col.to_numpy()[:cntXx[None]], hessianXx_val.to_numpy()[:cntXx[None]]
    # A = scipy.sparse.csr_matrix((val, (col, row)), shape=(n, n))
    # rows,cols = A.nonzero()
    # for row,col in zip(rows,cols):
    #     x_copy = x_bk.copy()
    #     x_copy[col // 3, col % 3] -= eps
    #     x0.from_numpy(x_copy)
    #     reset()
    #     data_rhs.fill(0)
    #     compute_gradient()
    #     g0 = - data_rhs[row]
    #     x_copy = x_bk.copy()
    #     x_copy[col // 3, col % 3] += eps
    #     x0.from_numpy(x_copy)
    #     reset()
    #     data_rhs.fill(0)
    #     compute_gradient()
    #     g1 = - data_rhs[row]
    #     print((g1 - g0) / (2 * eps), A[row, col])
    #     input()

def compute_hessian_and_gradient(pd=1):
    compute_gradient()
    compute_hessian(pd)

def solve_system(current_time):
    dirichlet_fixed, dirichlet_value = settings['dirichlet'](current_time)
    D = np.stack((dirichlet_fixed,) * dim, axis=-1).reshape((n_particles * dim))
    if cnt[None] >= MAX_LINEAR:
        print("FATAL ERROR: Array Too Small!")
    print("Total entries: ", cnt[None])
    # with Timer("DBC 0"):
        # dfx.from_numpy(D.astype(np.int32))
        # @ti.kernel
        # def DBC_set_zeros():
        #     for i in range(cnt[None]):
        #         if dfx[data_row[i]] or dfx[data_col[i]]:
        #             data_val[i] = 0
        # DBC_set_zeros()
    with Timer("Taichi to numpy"):
        row, col, val = data_row.to_numpy()[:cnt[None]], data_col.to_numpy()[:cnt[None]], data_val.to_numpy()[:cnt[None]]
        rhs = data_rhs.to_numpy()
    with Timer("DBC 1"):
        n = n_particles * dim
        A = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n))
        D = np.where(D)[0]
        A += scipy.sparse.csr_matrix((np.ones(len(D)), (D, D)), shape=(n, n))
        rhs[D] = 0
    with Timer("System Solve"):
        factor = cholesky(A)
        sol = factor(rhs)
    with Timer("Numpy to taichi"):
        data_sol.from_numpy(sol)


@ti.kernel
def save_x0():
    for i in range(n_particles):
        x0[i] = x[i]


@ti.kernel
def save_xPrev():
    for i in range(n_particles):
        xPrev[i] = x[i]


@ti.kernel
def apply_sol(alpha : real):
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            x(d)[i] = xPrev(d)[i] + data_sol[i * dim + d] * alpha


@ti.kernel
def compute_v():
    for i in range(n_particles):
        v[i] = (x[i] - xn[i]) / dt


@ti.kernel
def output_residual() -> real:
    residual = 0.0
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            residual = max(residual, ti.abs(data_sol[i * dim + d]))
    print("Search Direction Residual : ", residual / dt)
    return residual

mesh = None
if dim == 2:
    gui = ti.GUI("FEM", (768, 768), background_color=0x112F41)
# else:
#     scene = t3.Scene()
#     mesh = t3.DynamicMesh(n_faces=n_boundary_triangles, n_pos=n_particles)
#     model = t3.Model(mesh)  # here
#     scene.add_model(model)
#     camera = t3.Camera((768, 768))
#     scene.add_camera(camera)
#     light = t3.Light([0.4, -1.5, 1.8])
#     scene.add_light(light)
#     gui = ti.GUI('FEM', camera.res)

def write_image(f):
    particle_pos = x.to_numpy() * settings['mesh_scale'] + settings['mesh_offset']
    x_ = x.to_numpy()
    vertices_ = vertices.to_numpy()
    if dim == 2:
        for i in range(n_elements):
            for j in range(3):
                a, b = vertices_[i, j], vertices_[i, (j + 1) % 3]
                gui.line((particle_pos[a][0], particle_pos[a][1]),
                         (particle_pos[b][0], particle_pos[b][1]),
                         radius=1,
                         color=0x4FB99F)
        if 'visualize_segments' in settings:
            for a, b in settings['visualize_segments']:
                gui.line((particle_pos[a][0], particle_pos[a][1]),
                         (particle_pos[b][0], particle_pos[b][1]),
                         radius=1,
                         color=0xFFB99F)
        gui.show(directory + f'images/{f:06d}.png')
    else:
        # mesh.pos.from_numpy(particle_pos.astype(np.float32))
        # # mesh.faces.from_numpy(mesh_elements.astype(np.int32))
        # for e in range(n_elements):
        #     mesh.faces[e] = [[vertices[e, 0], 0, 0], [vertices[e, 1], 0, 0], [vertices[e, 2], 0, 0]]
        # camera.from_mouse(gui)
        # scene.render()
        # gui.set_image(camera.img)
        # gui.show(directory + f'images/{f:06d}.png')
        # f = open(directory + f'objs/{f:06d}.obj', 'w')
        # for i in range(n_particles):
        #     f.write('v %.6f %.6f %.6f\n' % (x_[i, 0], x_[i, 1], x_[i, 2]))
        # for [p0, p1, p2] in mesh_elements:
        #     f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
        # f.close()
        meshio.write_points_cells(
            directory + f'objs/{f:06d}.obj',
            particle_pos,
            [("triangle", mesh_elements)]
        )


if __name__ == "__main__":
    with Logger(directory + f'log.txt'):
        x0.from_numpy(mesh_particles.astype(np.float64))
        x.from_numpy(mesh_particles.astype(np.float64))
        v.fill(0)
        vertices.from_numpy(mesh_elements.astype(np.int32))
        edges.from_numpy(mesh_edges.astype(np.int32))
        init_material()
        reset()
        dirichlet_fixed, _ = settings['dirichlet'](0)
        D = np.stack((dirichlet_fixed,) * dim, axis=-1).reshape((n_particles * dim))
        dfx.from_numpy(D.astype(np.int32))
        if quasi_static:
            m.from_numpy(np.zeros((n_particles,)))
        # x.from_numpy(mesh_particles.astype(np.float64) * 2)
        zero.fill(0)
        write_image(0)
        f_start = 0
        if len(sys.argv) == 3:
            f_start = int(sys.argv[2])
            [x_, v_] = pickle.load(open(directory + f'caches/{f_start:06d}.p', 'rb'))
            x.from_numpy(x_)
            v.from_numpy(v_)
        newton_iter_total = 0
        current_time = f_start * dt
        for f in range(f_start, 120):
            with Timer("Time Step"):
                print("==================== Frame: ", f, " ====================")
                for step in range(sub_steps):
                    print("=============== Step: ", step, " =================")
                    compute_xn_and_xTilde()
                    move_nodes(current_time)
                    check_hessian_Xx()
                    # check_gradient()
                    # check_hessian()
                    # input()
                    newton_iter = 0
                    while True:
                        newton_iter += 1
                        print("-------------------- Newton Iteration: ", newton_iter, " --------------------")
                        with Timer("Build System"):
                            data_row.fill(0)
                            data_col.fill(0)
                            data_val.fill(0)
                            data_rhs.fill(0)
                            data_sol.fill(0)
                            compute_hessian_and_gradient()
                        with Timer("Solve System"):
                            solve_system(current_time)
                        if newton_iter > 1 and output_residual() < 1e-3 * dt:
                            break
                        if output_residual() < 1e-6 * dt:
                            break
                        with Timer("Line Search"):
                            E0 = compute_energy()
                            save_xPrev()
                            alpha = 1
                            apply_sol(alpha)
                            E = compute_energy()
                            while E > E0:
                                alpha *= 0.5
                                apply_sol(alpha)
                                E = compute_energy()
                            print("[Step size after line search: ", alpha, "]")
                    compute_v()
                    if quasi_static:
                        v.from_numpy(np.zeros((n_particles, 3)))
                    print("Edge Error: ", check_edge_error())
                    current_time += dt
                    newton_iter_total += newton_iter
                print("Avg Newton iter: ", newton_iter_total / (f + 1))
            with Timer("Visualization"):
                write_image(f + 1)
            pickle.dump([x.to_numpy(), v.to_numpy()], open(directory + f'caches/{f + 1:06d}.p', 'wb'))
            Timer_Print()
