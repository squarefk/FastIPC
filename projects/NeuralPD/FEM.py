from reader import *
from common.physics.fixed_corotated import *
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

##############################################################################
testcase = int(sys.argv[1])
settings = read()
mesh_particles = settings['mesh_particles']
mesh_elements = settings['mesh_elements']
dim = settings['dim']
gravity = settings['gravity']

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
ti.init(arch=ti.cpu, default_fp=real) #, cpu_max_num_threads=1)

scalar = lambda: ti.field(real)
vec = lambda: ti.Vector.field(dim, real)
mat = lambda: ti.Matrix.field(dim, dim, real)

dt = 0.04
n_particles = len(mesh_particles)
n_elements = len(mesh_elements)

x, x0, xPrev, xTilde, xn, v, m = vec(), vec(), vec(), vec(), vec(), vec(), scalar()
zero = vec()
la, mu = scalar(), scalar()
restT = mat()
B = mat()
vertices = ti.field(ti.i32)
ti.root.dense(ti.i, n_particles).place(x, x0, xPrev, xTilde, xn, v, m)
ti.root.dense(ti.i, n_particles).place(zero)
ti.root.dense(ti.i, n_elements).place(la, mu)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.i, n_elements).place(B)
ti.root.dense(ti.ij, (n_elements, dim + 1)).place(vertices)

MAX_LINEAR = 50000000 if dim == 3 else 5000000
data_rhs = ti.field(real, shape=n_particles * dim)
data_row = ti.field(ti.i32, shape=MAX_LINEAR)
data_col = ti.field(ti.i32, shape=MAX_LINEAR)
data_val = ti.field(real, shape=MAX_LINEAR)
data_sol = ti.field(real, shape=n_particles * dim)
cnt = ti.field(ti.i32, shape=())

dfx = ti.field(ti.i32, shape=n_particles * dim)


@ti.func
def compute_density(i):
    if ti.static(testcase == 1003):
        return 2000.0 if i < 1760 else 1000.0
    else:
        return 1000.0


@ti.func
def compute_lame_parameters(i):
    E = 0.0
    if ti.static(testcase == 1002):
        E = 2.e4
    elif ti.static(testcase == 1003):
        E = 1.e8 if i < 6851 else 1.e6
    else:
        E = 2.e4
    nu = 0.4
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
    if ti.static(dim == 2):
        ab = x[vertices[i, 1]] - x[vertices[i, 0]]
        ac = x[vertices[i, 2]] - x[vertices[i, 0]]
        return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])
    else:
        ab = x[vertices[i, 1]] - x[vertices[i, 0]]
        ac = x[vertices[i, 2]] - x[vertices[i, 0]]
        ad = x[vertices[i, 3]] - x[vertices[i, 0]]
        return ti.Matrix([[ab[0], ac[0], ad[0]], [ab[1], ac[1], ad[1]], [ab[2], ac[2], ad[2]]])


@ti.kernel
def compute_restT_and_m():
    for i in range(n_elements):
        restT[i] = compute_T(i)
        B[i] = restT[i].inverse()
        mass = restT[i].determinant() / dim / (dim - 1) * compute_density(i) / (dim + 1)
        if mass < 0.0:
            print("FATAL ERROR : mesh inverted")
        for d in ti.static(range(dim + 1)):
            m[vertices[i, d]] += mass
        la[i], mu[i] = compute_lame_parameters(i)


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
                x(d)[i] = dirichlet_value[i, d]
                xTilde(d)[i] = dirichlet_value[i, d]


@ti.kernel
def compute_energy() -> real:
    total_energy = 0.0
    # inertia
    for i in range(n_particles):
        total_energy += 0.5 * m[i] * (x[i] - xTilde[i]).norm_sqr()
    # elasticity
    for e in range(n_elements):
        F = compute_T(e) @ B[e]
        vol0 = restT[e].determinant() / dim / (dim - 1)
        U, sig, V = ti.svd(F)
        total_energy += elasticity_energy(sig, la[e], mu[e]) * dt * dt * vol0
    

@ti.func
def load_hessian_and_gradient(H, g, idx: ti.template(), c):
    for i in ti.static(range(idx.n)):
        for d in ti.static(range(dim)):
            for j in ti.static(range(idx.n)):
                for e in ti.static(range(dim)):
                    data_row[c], data_col[c], data_val[c] = idx[i] * dim + d, idx[j] * dim + e, H[i * dim + d, j * dim + e]
                    c += 1
    for i in ti.static(range(idx.n)):
        for d in ti.static(range(dim)):
            data_rhs[idx[i] * dim + d] -= g[i * dim + d]


@ti.kernel
def compute_inertia():
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            c = cnt[None] + i * dim + d
            data_row[c] = i * dim + d
            data_col[c] = i * dim + d
            data_val[c] = m[i]
            data_rhs[i * dim + d] -= m[i] * (x(d)[i] - xTilde(d)[i])
    cnt[None] += n_particles * dim


@ti.kernel
def compute_elasticity():
    for e in range(n_elements):
        F = compute_T(e) @ B[e]
        IB = B[e]
        vol0 = restT[e].determinant() / dim / (dim - 1)
        _la, _mu = la[e], mu[e]
        dPdF = elasticity_first_piola_kirchoff_stress_derivative(F, _la, _mu) * dt * dt * vol0
        P = elasticity_first_piola_kirchoff_stress(F, _la, _mu) * dt * dt * vol0
        if ti.static(dim == 2):
            intermediate = ti.Matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
            for colI in ti.static(range(4)):
                _000 = dPdF[0, colI] * IB[0, 0]
                _010 = dPdF[0, colI] * IB[1, 0]
                _101 = dPdF[2, colI] * IB[0, 1]
                _111 = dPdF[2, colI] * IB[1, 1]
                _200 = dPdF[1, colI] * IB[0, 0]
                _210 = dPdF[1, colI] * IB[1, 0]
                _301 = dPdF[3, colI] * IB[0, 1]
                _311 = dPdF[3, colI] * IB[1, 1]
                intermediate[2, colI] = _000 + _101
                intermediate[3, colI] = _200 + _301
                intermediate[4, colI] = _010 + _111
                intermediate[5, colI] = _210 + _311
                intermediate[0, colI] = -intermediate[2, colI] - intermediate[4, colI]
                intermediate[1, colI] = -intermediate[3, colI] - intermediate[5, colI]
            indMap = ti.Vector([vertices[e, 0] * 2, vertices[e, 0] * 2 + 1,
                                vertices[e, 1] * 2, vertices[e, 1] * 2 + 1,
                                vertices[e, 2] * 2, vertices[e, 2] * 2 + 1])
            for colI in ti.static(range(6)):
                _000 = intermediate[colI, 0] * IB[0, 0]
                _010 = intermediate[colI, 0] * IB[1, 0]
                _101 = intermediate[colI, 2] * IB[0, 1]
                _111 = intermediate[colI, 2] * IB[1, 1]
                _200 = intermediate[colI, 1] * IB[0, 0]
                _210 = intermediate[colI, 1] * IB[1, 0]
                _301 = intermediate[colI, 3] * IB[0, 1]
                _311 = intermediate[colI, 3] * IB[1, 1]
                c = cnt[None] + e * 36 + colI * 6 + 0
                data_row[c], data_col[c], data_val[c] = indMap[2], indMap[colI], _000 + _101
                c = cnt[None] + e * 36 + colI * 6 + 1
                data_row[c], data_col[c], data_val[c] = indMap[3], indMap[colI], _200 + _301
                c = cnt[None] + e * 36 + colI * 6 + 2
                data_row[c], data_col[c], data_val[c] = indMap[4], indMap[colI], _010 + _111
                c = cnt[None] + e * 36 + colI * 6 + 3
                data_row[c], data_col[c], data_val[c] = indMap[5], indMap[colI], _210 + _311
                c = cnt[None] + e * 36 + colI * 6 + 4
                data_row[c], data_col[c], data_val[c] = indMap[0], indMap[colI], - _000 - _101 - _010 - _111
                c = cnt[None] + e * 36 + colI * 6 + 5
                data_row[c], data_col[c], data_val[c] = indMap[1], indMap[colI], - _200 - _301 - _210 - _311
            data_rhs[vertices[e, 1] * 2 + 0] -= P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
            data_rhs[vertices[e, 1] * 2 + 1] -= P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
            data_rhs[vertices[e, 2] * 2 + 0] -= P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
            data_rhs[vertices[e, 2] * 2 + 1] -= P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
            data_rhs[vertices[e, 0] * 2 + 0] -= -P[0, 0] * IB[0, 0] - P[0, 1] * IB[0, 1] - P[0, 0] * IB[1, 0] - P[0, 1] * IB[1, 1]
            data_rhs[vertices[e, 0] * 2 + 1] -= -P[1, 0] * IB[0, 0] - P[1, 1] * IB[0, 1] - P[1, 0] * IB[1, 0] - P[1, 1] * IB[1, 1]
        else:
            Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            intermediate = ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z])
            for colI in ti.static(range(9)):
                intermediate[3, colI] = IB[0, 0] * dPdF[0, colI] + IB[0, 1] * dPdF[3, colI] + IB[0, 2] * dPdF[6, colI]
                intermediate[4, colI] = IB[0, 0] * dPdF[1, colI] + IB[0, 1] * dPdF[4, colI] + IB[0, 2] * dPdF[7, colI]
                intermediate[5, colI] = IB[0, 0] * dPdF[2, colI] + IB[0, 1] * dPdF[5, colI] + IB[0, 2] * dPdF[8, colI]
                intermediate[6, colI] = IB[1, 0] * dPdF[0, colI] + IB[1, 1] * dPdF[3, colI] + IB[1, 2] * dPdF[6, colI]
                intermediate[7, colI] = IB[1, 0] * dPdF[1, colI] + IB[1, 1] * dPdF[4, colI] + IB[1, 2] * dPdF[7, colI]
                intermediate[8, colI] = IB[1, 0] * dPdF[2, colI] + IB[1, 1] * dPdF[5, colI] + IB[1, 2] * dPdF[8, colI]
                intermediate[9, colI] = IB[2, 0] * dPdF[0, colI] + IB[2, 1] * dPdF[3, colI] + IB[2, 2] * dPdF[6, colI]
                intermediate[10, colI] = IB[2, 0] * dPdF[1, colI] + IB[2, 1] * dPdF[4, colI] + IB[2, 2] * dPdF[7, colI]
                intermediate[11, colI] = IB[2, 0] * dPdF[2, colI] + IB[2, 1] * dPdF[5, colI] + IB[2, 2] * dPdF[8, colI]
                intermediate[0, colI] = -intermediate[3, colI] - intermediate[6, colI] - intermediate[9, colI]
                intermediate[1, colI] = -intermediate[4, colI] - intermediate[7, colI] - intermediate[10, colI]
                intermediate[2, colI] = -intermediate[5, colI] - intermediate[8, colI] - intermediate[11, colI]
            indMap = ti.Vector([vertices[e, 0] * 3, vertices[e, 0] * 3 + 1, vertices[e, 0] * 3 + 2,
                                vertices[e, 1] * 3, vertices[e, 1] * 3 + 1, vertices[e, 1] * 3 + 2,
                                vertices[e, 2] * 3, vertices[e, 2] * 3 + 1, vertices[e, 2] * 3 + 2,
                                vertices[e, 3] * 3, vertices[e, 3] * 3 + 1, vertices[e, 3] * 3 + 2])
            for rowI in ti.static(range(12)):
                c = cnt[None] + e * 144 + rowI * 12 + 0
                data_row[c], data_col[c], data_val[c] = indMap[rowI], indMap[3], IB[0, 0] * intermediate[rowI, 0] + IB[0, 1] * intermediate[rowI, 3] + IB[0, 2] * intermediate[rowI, 6]
                c = cnt[None] + e * 144 + rowI * 12 + 1
                data_row[c], data_col[c], data_val[c] = indMap[rowI], indMap[4], IB[0, 0] * intermediate[rowI, 1] + IB[0, 1] * intermediate[rowI, 4] + IB[0, 2] * intermediate[rowI, 7]
                c = cnt[None] + e * 144 + rowI * 12 + 2
                data_row[c], data_col[c], data_val[c] = indMap[rowI], indMap[5], IB[0, 0] * intermediate[rowI, 2] + IB[0, 1] * intermediate[rowI, 5] + IB[0, 2] * intermediate[rowI, 8]
                c = cnt[None] + e * 144 + rowI * 12 + 3
                data_row[c], data_col[c], data_val[c] = indMap[rowI], indMap[6], IB[1, 0] * intermediate[rowI, 0] + IB[1, 1] * intermediate[rowI, 3] + IB[1, 2] * intermediate[rowI, 6]
                c = cnt[None] + e * 144 + rowI * 12 + 4
                data_row[c], data_col[c], data_val[c] = indMap[rowI], indMap[7], IB[1, 0] * intermediate[rowI, 1] + IB[1, 1] * intermediate[rowI, 4] + IB[1, 2] * intermediate[rowI, 7]
                c = cnt[None] + e * 144 + rowI * 12 + 5
                data_row[c], data_col[c], data_val[c] = indMap[rowI], indMap[8], IB[1, 0] * intermediate[rowI, 2] + IB[1, 1] * intermediate[rowI, 5] + IB[1, 2] * intermediate[rowI, 8]
                c = cnt[None] + e * 144 + rowI * 12 + 6
                data_row[c], data_col[c], data_val[c] = indMap[rowI], indMap[9], IB[2, 0] * intermediate[rowI, 0] + IB[2, 1] * intermediate[rowI, 3] + IB[2, 2] * intermediate[rowI, 6]
                c = cnt[None] + e * 144 + rowI * 12 + 7
                data_row[c], data_col[c], data_val[c] = indMap[rowI], indMap[10], IB[2, 0] * intermediate[rowI, 1] + IB[2, 1] * intermediate[rowI, 4] + IB[2, 2] * intermediate[rowI, 7]
                c = cnt[None] + e * 144 + rowI * 12 + 8
                data_row[c], data_col[c], data_val[c] = indMap[rowI], indMap[11], IB[2, 0] * intermediate[rowI, 2] + IB[2, 1] * intermediate[rowI, 5] + IB[2, 2] * intermediate[rowI, 8]
                c = cnt[None] + e * 144 + rowI * 12 + 9
                data_row[c], data_col[c], data_val[c] = indMap[rowI], indMap[0], -data_val[c - 9] - data_val[c - 6] - data_val[c - 3]
                c = cnt[None] + e * 144 + rowI * 12 + 10
                data_row[c], data_col[c], data_val[c] = indMap[rowI], indMap[1], -data_val[c - 9] - data_val[c - 6] - data_val[c - 3]
                c = cnt[None] + e * 144 + rowI * 12 + 11
                data_row[c], data_col[c], data_val[c] = indMap[rowI], indMap[2], -data_val[c - 9] - data_val[c - 6] - data_val[c - 3]
            R10 = IB[0, 0] * P[0, 0] + IB[0, 1] * P[0, 1] + IB[0, 2] * P[0, 2]
            R11 = IB[0, 0] * P[1, 0] + IB[0, 1] * P[1, 1] + IB[0, 2] * P[1, 2]
            R12 = IB[0, 0] * P[2, 0] + IB[0, 1] * P[2, 1] + IB[0, 2] * P[2, 2]
            R20 = IB[1, 0] * P[0, 0] + IB[1, 1] * P[0, 1] + IB[1, 2] * P[0, 2]
            R21 = IB[1, 0] * P[1, 0] + IB[1, 1] * P[1, 1] + IB[1, 2] * P[1, 2]
            R22 = IB[1, 0] * P[2, 0] + IB[1, 1] * P[2, 1] + IB[1, 2] * P[2, 2]
            R30 = IB[2, 0] * P[0, 0] + IB[2, 1] * P[0, 1] + IB[2, 2] * P[0, 2]
            R31 = IB[2, 0] * P[1, 0] + IB[2, 1] * P[1, 1] + IB[2, 2] * P[1, 2]
            R32 = IB[2, 0] * P[2, 0] + IB[2, 1] * P[2, 1] + IB[2, 2] * P[2, 2]
            data_rhs[vertices[e, 1] * 3 + 0] -= R10
            data_rhs[vertices[e, 1] * 3 + 1] -= R11
            data_rhs[vertices[e, 1] * 3 + 2] -= R12
            data_rhs[vertices[e, 2] * 3 + 0] -= R20
            data_rhs[vertices[e, 2] * 3 + 1] -= R21
            data_rhs[vertices[e, 2] * 3 + 2] -= R22
            data_rhs[vertices[e, 3] * 3 + 0] -= R30
            data_rhs[vertices[e, 3] * 3 + 1] -= R31
            data_rhs[vertices[e, 3] * 3 + 2] -= R32
            data_rhs[vertices[e, 0] * 3 + 0] -= -R10 - R20 - R30
            data_rhs[vertices[e, 0] * 3 + 1] -= -R11 - R21 - R31
            data_rhs[vertices[e, 0] * 3 + 2] -= -R12 - R22 - R32
    cnt[None] += n_elements * (dim + 1) * dim * (dim + 1) * dim


def compute_hessian_and_gradient():
    cnt[None] = 0
    compute_inertia()
    compute_elasticity()

def solve_system(current_time):
    dirichlet_fixed, dirichlet_value = settings['dirichlet'](current_time)
    D, V = np.stack((dirichlet_fixed,) * dim, axis=-1).reshape((n_particles * dim)), np.zeros((n_particles * dim))
    if cnt[None] >= MAX_LINEAR:
        print("FATAL ERROR: Array Too Small!")
    print("Total entries: ", cnt[None])
    with Timer("DBC 0"):
        dfx.from_numpy(D.astype(np.int32))
        @ti.kernel
        def DBC_set_zeros():
            for i in range(cnt[None]):
                if dfx[data_row[i]] or dfx[data_col[i]]:
                    data_val[i] = 0
        DBC_set_zeros()
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


if dim == 2:
    gui = ti.GUI("FEM", (768, 768), background_color=0x112F41)
else:
    scene = t3.Scene()
    model = t3.Model(f_n=n_boundary_triangles, vi_n=n_particles)
    scene.add_model(model)
    camera = t3.Camera((768, 768))
    scene.add_camera(camera)
    light = t3.Light([0.4, -1.5, 1.8])
    scene.add_light(light)
    gui = ti.GUI('FEM', camera.res)

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
        model.vi.from_numpy(particle_pos.astype(np.float32))
        model.faces.from_numpy(boundary_triangles_.astype(np.int32))
        camera.from_mouse(gui)
        scene.render()
        gui.set_image(camera.img)
        gui.show(directory + f'images/{f:06d}.png')
        f = open(directory + f'objs/{f:06d}.obj', 'w')
        for i in range(n_particles):
            f.write('v %.6f %.6f %.6f\n' % (x_[i, 0], x_[i, 1], x_[i, 2]))
        for [p0, p1, p2] in boundary_triangles_:
            f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
        f.close()


if __name__ == "__main__":
    with Logger(directory + f'log.txt'):
        x.from_numpy(mesh_particles.astype(np.float64))
        v.fill(0)
        vertices.from_numpy(mesh_elements.astype(np.int32))
        compute_restT_and_m()
        save_x0()
        zero.fill(0)
        write_image(0)
        f_start = 0
        if len(sys.argv) == 3:
            f_start = int(sys.argv[2])
            [x_, v_] = pickle.load(open(directory + f'caches/{f_start:06d}.p', 'rb'))
            x.from_numpy(x_)
            v.from_numpy(v_)
        newton_iter_total = 0
        for f in range(f_start, 10000):
            with Timer("Time Step"):
                print("==================== Frame: ", f, " ====================")
                compute_xn_and_xTilde()
                move_nodes(f * dt)
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
                        solve_system(f * dt)
                    if output_residual() < 1e-2 * dt:
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
                            find_constraints()
                            E = compute_energy()
                        print("[Step size after line search: ", alpha, "]")
                compute_v()
                newton_iter_total += newton_iter
                print("Avg Newton iter: ", newton_iter_total / (f + 1))
            with Timer("Visualization"):
                write_image(f + 1)
            pickle.dump([x.to_numpy(), v.to_numpy()], open(directory + f'caches/{f + 1:06d}.p', 'wb'))
            Timer_Print()
