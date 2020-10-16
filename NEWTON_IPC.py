import sys, os, time, math
import taichi as ti
import taichi_three as t3
import numpy as np
from neo_hookean_3d import *
from math_tools import *
from ipc import *
import meshio
import scipy.sparse
import scipy.sparse.linalg
from reader import *

##############################################################################

mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, dim = read(int(sys.argv[1]))
triangles = set()
for [p0, p1, p2, p3] in mesh_elements:
    triangles.add((p0, p2, p1))
    triangles.add((p0, p3, p2))
    triangles.add((p0, p1, p3))
    triangles.add((p1, p2, p3))
boundary_points_ = set()
boundary_edges_ = np.zeros(shape=(0, 2), dtype=np.int32)
boundary_triangles_ = np.zeros(shape=(0, 3), dtype=np.int32)
for (p0, p1, p2) in triangles:
    if (p0, p2, p1) not in triangles:
        if (p2, p1, p0) not in triangles:
            if (p1, p0, p2) not in triangles:
                boundary_points_.update([p0, p1, p2])
                if p0 < p1:
                    boundary_edges_ = np.vstack((boundary_edges_, [p0, p1]))
                if p1 < p2:
                    boundary_edges_ = np.vstack((boundary_edges_, [p1, p2]))
                if p2 < p0:
                    boundary_edges_ = np.vstack((boundary_edges_, [p2, p0]))
                boundary_triangles_ = np.vstack((boundary_triangles_, [p0, p1, p2]))

##############################################################################

directory = 'output/' + '_'.join(sys.argv) + '/'
os.makedirs(directory + 'images/', exist_ok=True)
print('output directory:', directory)
# sys.stdout = open(directory + 'log.txt', 'w')
# sys.stderr = open(directory + 'err.txt', 'w')

##############################################################################

real = ti.f64
ti.init(arch=ti.cpu, default_fp=real)

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

dt = 0.01
E = 2e4
nu = 0.4
la = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
density = 1000
n_particles = len(mesh_particles)
n_elements = len(mesh_elements)
n_boundary_points = len(boundary_points_)
n_boundary_edges = len(boundary_edges_)
n_boundary_triangles = len(boundary_triangles_)

x, x0, xPrev, xTilde, xn, v, m = vec(), vec(), vec(), vec(), vec(), vec(), scalar()
zero = vec()
restT = mat()
vertices = ti.var(ti.i32)
boundary_points = ti.var(ti.i32)
boundary_edges = ti.var(ti.i32)
boundary_triangles = ti.var(ti.i32)
dirichlet = ti.var(ti.i32)
ti.root.dense(ti.k, n_particles).place(x, x0, xPrev, xTilde, xn, v, m)
ti.root.dense(ti.k, n_particles).place(zero)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.ij, (n_elements, dim + 1)).place(vertices)
ti.root.dense(ti.i, n_boundary_points).place(boundary_points)
ti.root.dense(ti.ij, (n_boundary_edges, 2)).place(boundary_edges)
ti.root.dense(ti.ij, (n_boundary_triangles, 3)).place(boundary_triangles)
ti.root.dense(ti.i, n_particles * dim).place(dirichlet)

data_rhs = ti.var(real, shape=n_particles * dim)
data_row = ti.var(ti.i32, shape=10000000)
data_col = ti.var(ti.i32, shape=10000000)
data_val = ti.var(real, shape=10000000)
data_sol = ti.var(real, shape=n_particles * dim)
cnt = ti.var(dt=ti.i32, shape=())

PP = ti.var(ti.i32, shape=(100000, 2))
n_PP = ti.var(dt=ti.i32, shape=())
PE = ti.var(ti.i32, shape=(100000, 3))
n_PE = ti.var(dt=ti.i32, shape=())
PT = ti.var(ti.i32, shape=(100000, 4))
n_PT = ti.var(dt=ti.i32, shape=())
EE = ti.var(ti.i32, shape=(100000, 4))
n_EE = ti.var(dt=ti.i32, shape=())
EEM = ti.var(ti.i32, shape=(100000, 4))
n_EEM = ti.var(dt=ti.i32, shape=())
PPM = ti.var(ti.i32, shape=(100000, 4))
n_PPM = ti.var(dt=ti.i32, shape=())
PEM = ti.var(ti.i32, shape=(100000, 4))
n_PEM = ti.var(dt=ti.i32, shape=())

dHat2 = 1e-5
dHat = dHat2 ** 0.5
kappa = 1e4


@ti.kernel
def find_constraints():
    n_PP[None], n_PE[None], n_PT[None], n_EE[None], n_EEM[None], n_PPM[None], n_PEM[None] = 0, 0, 0, 0, 0, 0, 0
    for i in range(n_boundary_points):
        p = boundary_points[i]
        for j in range(n_boundary_triangles):
            t0 = boundary_triangles[j, 0]
            t1 = boundary_triangles[j, 1]
            t2 = boundary_triangles[j, 2]
            if p != t0 and p != t1 and p != t2 and point_triangle_ccd_broadphase(x[p], x[t0], x[t1], x[t2], dHat):
                case = PT_type(x[p], x[t0], x[t1], x[t2])
                if case == 0:
                    if PP_3D_E(x[p], x[t0]) < dHat2:
                        n = ti.atomic_add(n_PP[None], 1)
                        PP[n, 0], PP[n, 1] = p, t0
                elif case == 1:
                    if PP_3D_E(x[p], x[t1]) < dHat2:
                        n = ti.atomic_add(n_PP[None], 1)
                        PP[n, 0], PP[n, 1] = p, t1
                elif case == 2:
                    if PP_3D_E(x[p], x[t2]) < dHat2:
                        n = ti.atomic_add(n_PP[None], 1)
                        PP[n, 0], PP[n, 1] = p, t2
                elif case == 3:
                    if PE_3D_E(x[p], x[t0], x[t1]) < dHat2:
                        n = ti.atomic_add(n_PE[None], 1)
                        PE[n, 0], PE[n, 1], PE[n, 2] = p, t0, t1
                elif case == 4:
                    if PE_3D_E(x[p], x[t1], x[t2]) < dHat2:
                        n = ti.atomic_add(n_PE[None], 1)
                        PE[n, 0], PE[n, 1], PE[n, 2] = p, t1, t2
                elif case == 5:
                    if PE_3D_E(x[p], x[t2], x[t0]) < dHat2:
                        n = ti.atomic_add(n_PE[None], 1)
                        PE[n, 0], PE[n, 1], PE[n, 2] = p, t2, t0
                elif case == 6:
                    if PT_3D_E(x[p], x[t0], x[t1], x[t2]) < dHat2:
                        n = ti.atomic_add(n_PT[None], 1)
                        PT[n, 0], PT[n, 1], PT[n, 2], PT[n, 3] = p, t0, t1, t2
    for i in range(n_boundary_edges):
        a0 = boundary_edges[i, 0]
        a1 = boundary_edges[i, 1]
        for j in range(n_boundary_edges):
            b0 = boundary_edges[j, 0]
            b1 = boundary_edges[j, 1]
            if i < j and a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1 and edge_edge_ccd_broadphase(x[a0], x[a1], x[b0], x[b1], dHat):
                EECN2 = EECN2_E(x[a0], x[a1], x[b0], x[b1])
                eps_x = M_threshold(x0[a0], x0[a1], x0[b0], x0[b1])
                case = EE_type(x[a0], x[a1], x[b0], x[b1])
                if case == 0:
                    if PP_3D_E(x[a0], x[b0]) < dHat2:
                        if EECN2 < eps_x:
                            n = ti.atomic_add(n_PPM[None], 1)
                            PPM[n, 0], PPM[n, 1], PPM[n, 2], PPM[n, 3] = a0, a1, b0, b1
                        else:
                            n = ti.atomic_add(n_PP[None], 1)
                            PP[n, 0], PP[n, 1] = a0, b0
                elif case == 1:
                    if PP_3D_E(x[a0], x[b1]) < dHat2:
                        if EECN2 < eps_x:
                            n = ti.atomic_add(n_PPM[None], 1)
                            PPM[n, 0], PPM[n, 1], PPM[n, 2], PPM[n, 3] = a0, a1, b1, b0
                        else:
                            n = ti.atomic_add(n_PP[None], 1)
                            PP[n, 0], PP[n, 1] = a0, b1
                elif case == 2:
                    if PE_3D_E(x[a0], x[b0], x[b1]) < dHat2:
                        if EECN2 < eps_x:
                            n = ti.atomic_add(n_PEM[None], 1)
                            PEM[n, 0], PEM[n, 1], PEM[n, 2], PEM[n, 3] = a0, a1, b0, b1
                        else:
                            n = ti.atomic_add(n_PE[None], 1)
                            PE[n, 0], PE[n, 1], PE[n, 2] = a0, b0, b1
                elif case == 3:
                    if PP_3D_E(x[a1], x[b0]) < dHat2:
                        if EECN2 < eps_x:
                            n = ti.atomic_add(n_PPM[None], 1)
                            PPM[n, 0], PPM[n, 1], PPM[n, 2], PPM[n, 3] = a1, a0, b0, b1
                        else:
                            n = ti.atomic_add(n_PP[None], 1)
                            PP[n, 0], PP[n, 1] = a1, b0
                elif case == 4:
                    if PP_3D_E(x[a1], x[b1]) < dHat2:
                        if EECN2 < eps_x:
                            n = ti.atomic_add(n_PPM[None], 1)
                            PPM[n, 0], PPM[n, 1], PPM[n, 2], PPM[n, 3] = a1, a0, b1, b0
                        else:
                            n = ti.atomic_add(n_PP[None], 1)
                            PP[n, 0], PP[n, 1] = a1, b1
                elif case == 5:
                    if PE_3D_E(x[a1], x[b0], x[b1]) < dHat2:
                        if EECN2 < eps_x:
                            n = ti.atomic_add(n_PEM[None], 1)
                            PEM[n, 0], PEM[n, 1], PEM[n, 2], PEM[n, 3] = a1, a0, b0, b1
                        else:
                            n = ti.atomic_add(n_PE[None], 1)
                            PE[n, 0], PE[n, 1], PE[n, 2] = a1, b0, b1
                elif case == 6:
                    if PE_3D_E(x[b0], x[a0], x[a1]) < dHat2:
                        if EECN2 < eps_x:
                            n = ti.atomic_add(n_PEM[None], 1)
                            PEM[n, 0], PEM[n, 1], PEM[n, 2], PEM[n, 3] = b0, b1, a0, a1
                        else:
                            n = ti.atomic_add(n_PE[None], 1)
                            PE[n, 0], PE[n, 1], PE[n, 2] = b0, a0, a1
                elif case == 7:
                    if PE_3D_E(x[b1], x[a0], x[a1]) < dHat2:
                        if EECN2 < eps_x:
                            n = ti.atomic_add(n_PEM[None], 1)
                            PEM[n, 0], PEM[n, 1], PEM[n, 2], PEM[n, 3] = b1, b0, a0, a1
                        else:
                            n = ti.atomic_add(n_PE[None], 1)
                            PE[n, 0], PE[n, 1], PE[n, 2] = b1, a0, a1
                elif case == 8:
                    if EE_3D_E(x[a0], x[a1], x[b0], x[b1]) < dHat2:
                        if EECN2 < eps_x:
                            n = ti.atomic_add(n_EEM[None], 1)
                            EEM[n, 0], EEM[n, 1], EEM[n, 2], EEM[n, 3] = a0, a1, b0, b1
                        else:
                            n = ti.atomic_add(n_EE[None], 1)
                            EE[n, 0], EE[n, 1], EE[n, 2], EE[n, 3] = a0, a1, b0, b1
    print("Find constraints: ", n_PP[None], n_PE[None], n_PT[None], n_EE[None], n_EEM[None], n_PPM[None], n_PEM[None])


@ti.kernel
def compute_intersection_free_step_size() -> real:
    alpha = 1.0
    for i in range(n_boundary_points):
        p = boundary_points[i]
        for j in range(n_boundary_triangles):
            t0 = boundary_triangles[j, 0]
            t1 = boundary_triangles[j, 1]
            t2 = boundary_triangles[j, 2]
            if p != t0 and p != t1 and p != t2:
                dp = ti.Vector([data_sol[p * dim + 0], data_sol[p * dim + 1], data_sol[p * dim + 2]])
                dt0 = ti.Vector([data_sol[t0 * dim + 0], data_sol[t0 * dim + 1], data_sol[t0 * dim + 2]])
                dt1 = ti.Vector([data_sol[t1 * dim + 0], data_sol[t1 * dim + 1], data_sol[t1 * dim + 2]])
                dt2 = ti.Vector([data_sol[t2 * dim + 0], data_sol[t2 * dim + 1], data_sol[t2 * dim + 2]])
                if moving_point_triangle_ccd_broadphase(x[p], x[t0], x[t1], x[t2], dp, dt0, dt1, dt2, dHat):
                    dist2 = PT_dist2(x[p], x[t0], x[t1], x[t2], PT_type(x[p], x[t0], x[t1], x[t2]))
                    alpha = ti.min(alpha, point_triangle_ccd(x[p], x[t0], x[t1], x[t2], dp, dt0, dt1, dt2, 0.2, dist2))
    for i in range(n_boundary_edges):
        a0 = boundary_edges[i, 0]
        a1 = boundary_edges[i, 1]
        for j in range(n_boundary_edges):
            b0 = boundary_edges[j, 0]
            b1 = boundary_edges[j, 1]
            if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1:
                da0 = ti.Vector([data_sol[a0 * dim + 0], data_sol[a0 * dim + 1], data_sol[a0 * dim + 2]])
                da1 = ti.Vector([data_sol[a1 * dim + 0], data_sol[a1 * dim + 1], data_sol[a1 * dim + 2]])
                db0 = ti.Vector([data_sol[b0 * dim + 0], data_sol[b0 * dim + 1], data_sol[b0 * dim + 2]])
                db1 = ti.Vector([data_sol[b1 * dim + 0], data_sol[b1 * dim + 1], data_sol[b1 * dim + 2]])
                if moving_edge_edge_ccd_broadphase(x[a0], x[a1], x[b0], x[b1], da0, da1, db0, db1, dHat):
                    dist2 = EE_dist2(x[a0], x[a1], x[b0], x[b1], EE_type(x[a0], x[a1], x[b0], x[b1]))
                    alpha = ti.min(alpha, edge_edge_ccd(x[a0], x[a1], x[b0], x[b1], da0, da1, db0, db1, 0.2, dist2))
    for i in range(n_elements):
        a, b, c, d = vertices[i, 0], vertices[i, 1], vertices[i, 2], vertices[i, 3]
        da = ti.Vector([data_sol[a * dim + 0], data_sol[a * dim + 1], data_sol[a * dim + 2]])
        db = ti.Vector([data_sol[b * dim + 0], data_sol[b * dim + 1], data_sol[b * dim + 2]])
        dc = ti.Vector([data_sol[c * dim + 0], data_sol[c * dim + 1], data_sol[c * dim + 2]])
        dd = ti.Vector([data_sol[d * dim + 0], data_sol[d * dim + 1], data_sol[d * dim + 2]])
        alpha = ti.min(alpha, get_smallest_positive_real_cubic_root(x[a], x[b], x[c], x[d], da, db, dc, dd, 0.2))
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
        mass = restT[i].determinant() / dim / (dim - 1) * density / (dim + 1)
        if mass < 0.0:
            print("FATAL ERROR : mesh inverted")
        for d in ti.static(range(dim + 1)):
            m[vertices[i, d]] += mass


@ti.kernel
def compute_xn_and_xTilde():
    for i in range(n_particles):
        xn[i] = x[i]
        xTilde[i] = x[i] + dt * v[i]
        # xTilde(0)[i] -= dt * dt * 9.8


@ti.kernel
def move_nodes():
    speed = math.pi / 100
    for i in range(n_particles):
        if dirichlet[i * dim]:
            a, b, c = x(0)[i], x(1)[i], x(2)[i]
            angle = ti.atan2(b, c)
            if a < 0:
                angle += speed
            else:
                angle -= speed
            radius = ti.sqrt(b * b + c * c)
            x(1)[i], x(2)[i] = radius * ti.sin(angle), radius * ti.cos(angle)


@ti.kernel
def compute_energy() -> real:
    total_energy = 0.0
    # inertia
    for i in range(n_particles):
        total_energy += 0.5 * m[i] * (x[i] - xTilde[i]).norm_sqr()
    # elasticity
    for e in range(n_elements):
        F = compute_T(e) @ restT[e].inverse()
        vol0 = restT[e].determinant() / dim / (dim - 1)
        U, sig, V = ti.svd(F)
        total_energy += elasticity_energy(sig, la, mu) * dt * dt * vol0
    # ipc
    for r in range(n_PP[None]):
        p0, p1 = x[PP[r, 0]], x[PP[r, 1]]
        dist2 = PP_3D_E(p0, p1)
        total_energy += barrier_E(dist2, dHat2, kappa)
    for r in range(n_PE[None]):
        p, e0, e1 = x[PE[r, 0]], x[PE[r, 1]], x[PE[r, 2]]
        dist2 = PE_3D_E(p, e0, e1)
        total_energy += barrier_E(dist2, dHat2, kappa)
    for r in range(n_PT[None]):
        p, t0, t1, t2 = x[PT[r, 0]], x[PT[r, 1]], x[PT[r, 2]], x[PT[r, 3]]
        dist2 = PT_3D_E(p, t0, t1, t2)
        total_energy += barrier_E(dist2, dHat2, kappa)
    for r in range(n_EE[None]):
        a0, a1, b0, b1 = x[EE[r, 0]], x[EE[r, 1]], x[EE[r, 2]], x[EE[r, 3]]
        dist2 = EE_3D_E(a0, a1, b0, b1)
        total_energy += barrier_E(dist2, dHat2, kappa)
    for r in range(n_EEM[None]):
        a0, a1, b0, b1 = x[EEM[r, 0]], x[EEM[r, 1]], x[EEM[r, 2]], x[EEM[r, 3]]
        _a0, _a1, _b0, _b1 = x0[EEM[r, 0]], x0[EEM[r, 1]], x0[EEM[r, 2]], x0[EEM[r, 3]]
        dist2 = EE_3D_E(a0, a1, b0, b1)
        eps_x = M_threshold(_a0, _a1, _b0, _b1)
        total_energy += barrier_E(dist2, dHat2, kappa) * M_E(a0, a1, b0, b1, eps_x)
    for r in range(n_PPM[None]):
        a0, a1, b0, b1 = x[PPM[r, 0]], x[PPM[r, 1]], x[PPM[r, 2]], x[PPM[r, 3]]
        _a0, _a1, _b0, _b1 = x0[PPM[r, 0]], x0[PPM[r, 1]], x0[PPM[r, 2]], x0[PPM[r, 3]]
        dist2 = PP_3D_E(a0, b0)
        eps_x = M_threshold(_a0, _a1, _b0, _b1)
        total_energy += barrier_E(dist2, dHat2, kappa) * M_E(a0, a1, b0, b1, eps_x)
    for r in range(n_PEM[None]):
        a0, a1, b0, b1 = x[PEM[r, 0]], x[PEM[r, 1]], x[PEM[r, 2]], x[PEM[r, 3]]
        _a0, _a1, _b0, _b1 = x0[PEM[r, 0]], x0[PEM[r, 1]], x0[PEM[r, 2]], x0[PEM[r, 3]]
        dist2 = PE_3D_E(a0, b0, b1)
        eps_x = M_threshold(_a0, _a1, _b0, _b1)
        total_energy += barrier_E(dist2, dHat2, kappa) * M_E(a0, a1, b0, b1, eps_x)
    return total_energy


@ti.func
def load_hessian_and_gradient(H, g, idx, c, t: ti.template()):
    for i in ti.static(range(t)):
        for d in ti.static(range(dim)):
            for j in ti.static(range(t)):
                for e in ti.static(range(dim)):
                    data_row[c], data_col[c], data_val[c] = idx[i] * dim + d, idx[j] * dim + e, H[i * dim + d, j * dim + e]
                    c += 1
    for i in ti.static(range(t)):
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
    print("Start Doing!")
    for e in range(n_elements):
        F = compute_T(e) @ restT[e].inverse()
        IB = restT[e].inverse()
        vol0 = restT[e].determinant() / dim / (dim - 1)
        dPdF = elasticity_first_piola_kirchoff_stress_derivative(F, la, mu) * dt * dt * vol0
        P = elasticity_first_piola_kirchoff_stress(F, la, mu) * dt * dt * vol0
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
@ti.kernel
def compute_ipc0():
    for r in range(n_PP[None]):
        p0, p1 = x[PP[r, 0]], x[PP[r, 1]]
        dist2 = PP_3D_E(p0, p1)
        dist2g = PP_3D_g(p0, p1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_3D_H(p0, p1)
        load_hessian_and_gradient(H, g, ti.Vector([PP[r, 0], PP[r, 1]]), cnt[None] + r * 36, 2)
    cnt[None] += n_PP[None] * 36
@ti.kernel
def compute_ipc1():
    for r in range(n_PE[None]):
        p, e0, e1 = x[PE[r, 0]], x[PE[r, 1]], x[PE[r, 2]]
        dist2 = PE_3D_E(p, e0, e1)
        dist2g = PE_3D_g(p, e0, e1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_3D_H(p, e0, e1)
        load_hessian_and_gradient(H, g, ti.Vector([PE[r, 0], PE[r, 1], PE[r, 2]]), cnt[None] + r * 81, 3)
    cnt[None] += n_PE[None] * 81
@ti.kernel
def compute_ipc2():
    for r in range(n_PT[None]):
        p, t0, t1, t2 = x[PT[r, 0]], x[PT[r, 1]], x[PT[r, 2]], x[PT[r, 3]]
        dist2 = PT_3D_E(p, t0, t1, t2)
        dist2g = PT_3D_g(p, t0, t1, t2)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PT_3D_H(p, t0, t1, t2)
        load_hessian_and_gradient(H, g, ti.Vector([PT[r, 0], PT[r, 1], PT[r, 2], PT[r, 3]]), cnt[None] + r * 144, 4)
    cnt[None] += n_PT[None] * 144
@ti.kernel
def compute_ipc3():
    for r in range(n_EE[None]):
        a0, a1, b0, b1 = x[EE[r, 0]], x[EE[r, 1]], x[EE[r, 2]], x[EE[r, 3]]
        dist2 = EE_3D_E(a0, a1, b0, b1)
        dist2g = EE_3D_g(a0, a1, b0, b1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * EE_3D_H(a0, a1, b0, b1)
        load_hessian_and_gradient(H, g, ti.Vector([EE[r, 0], EE[r, 1], EE[r, 2], EE[r, 3]]), cnt[None] + r * 144, 4)
    cnt[None] += n_EE[None] * 144
@ti.kernel
def compute_ipc4():
    for r in range(n_EEM[None]):
        a0, a1, b0, b1 = x[EEM[r, 0]], x[EEM[r, 1]], x[EEM[r, 2]], x[EEM[r, 3]]
        _a0, _a1, _b0, _b1 = x0[EEM[r, 0]], x0[EEM[r, 1]], x0[EEM[r, 2]], x0[EEM[r, 3]]
        eps_x = M_threshold(_a0, _a1, _b0, _b1)
        dist2 = EE_3D_E(a0, a1, b0, b1)
        dist2g = EE_3D_g(a0, a1, b0, b1)
        b = barrier_E(dist2, dHat2, kappa)
        bg = barrier_g(dist2, dHat2, kappa)
        lg = bg * dist2g
        lH = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * EE_3D_H(a0, a1, b0, b1)
        M = M_E(a0, a1, b0, b1, eps_x)
        Mg = M_g(a0, a1, b0, b1, eps_x)
        g = lg * M + b * Mg
        H = lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
        load_hessian_and_gradient(H, g, ti.Vector([EEM[r, 0], EEM[r, 1], EEM[r, 2], EEM[r, 3]]), cnt[None] + r * 144, 4)
    cnt[None] += n_EEM[None] * 144
@ti.kernel
def compute_ipc5():
    for r in range(n_PPM[None]):
        a0, a1, b0, b1 = x[PPM[r, 0]], x[PPM[r, 1]], x[PPM[r, 2]], x[PPM[r, 3]]
        _a0, _a1, _b0, _b1 = x0[PPM[r, 0]], x0[PPM[r, 1]], x0[PPM[r, 2]], x0[PPM[r, 3]]
        eps_x = M_threshold(_a0, _a1, _b0, _b1)
        dist2 = PP_3D_E(a0, b0)
        dist2g = PP_3D_g(a0, b0)
        b = barrier_E(dist2, dHat2, kappa)
        bg = barrier_g(dist2, dHat2, kappa)
        idx = [0, 1, 2, 6, 7, 8]
        lg = fill_vec(bg * dist2g, idx)
        lH = fill_mat(barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_3D_H(a0, b0), idx)
        M = M_E(a0, a1, b0, b1, eps_x)
        Mg = M_g(a0, a1, b0, b1, eps_x)
        g = lg * M + b * Mg
        H = lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
        load_hessian_and_gradient(H, g, ti.Vector([PPM[r, 0], PPM[r, 1], PPM[r, 2], PPM[r, 3]]), cnt[None] + r * 144, 4)
    cnt[None] += n_PPM[None] * 144
@ti.kernel
def compute_ipc6():
    for r in range(n_PEM[None]):
        a0, a1, b0, b1 = x[PEM[r, 0]], x[PEM[r, 1]], x[PEM[r, 2]], x[PEM[r, 3]]
        _a0, _a1, _b0, _b1 = x0[PEM[r, 0]], x0[PEM[r, 1]], x0[PEM[r, 2]], x0[PEM[r, 3]]
        eps_x = M_threshold(_a0, _a1, _b0, _b1)
        dist2 = PE_3D_E(a0, b0, b1)
        dist2g = PE_3D_g(a0, b0, b1)
        b = barrier_E(dist2, dHat2, kappa)
        bg = barrier_g(dist2, dHat2, kappa)
        idx = [0, 1, 2, 6, 7, 8, 9, 10, 11]
        lg = fill_vec(bg * dist2g, idx)
        lH = fill_mat(barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_3D_H(a0, b0, b1), idx)
        M = M_E(a0, a1, b0, b1, eps_x)
        Mg = M_g(a0, a1, b0, b1, eps_x)
        g = lg * M + b * Mg
        H = lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
        load_hessian_and_gradient(H, g, ti.Vector([PEM[r, 0], PEM[r, 1], PEM[r, 2], PEM[r, 3]]), cnt[None] + r * 144, 4)
    cnt[None] += n_PEM[None] * 144


def compute_hessian_and_gradient():
    cnt[None] = 0
    print("Start computing H and g.", end='')
    compute_inertia()
    print("inertia done.", end='')
    compute_elasticity()
    print("elasticity done")
    compute_ipc0()
    print("ipc0 done.", end='')
    compute_ipc1()
    print("ipc1 done.", end='')
    compute_ipc2()
    print("ipc2 done.", end='')
    compute_ipc3()
    print("ipc3 done.", end='')
    compute_ipc4()
    print("ipc4 done.", end='')
    compute_ipc5()
    print("ipc5 done.", end='')
    compute_ipc6()
    print("ipc6 done.", end='\n')


def solve_system():
    print("Total entries: ", cnt[None])
    row, col, val = data_row.to_numpy()[:cnt[None]], data_col.to_numpy()[:cnt[None]], data_val.to_numpy()[:cnt[None]]
    rhs = data_rhs.to_numpy()
    c36 = n_PP[None]
    c81 = n_PE[None]
    c144 = n_PT[None] + n_EE[None] + n_EEM[None] + n_PPM[None] + n_PEM[None]
    for i in range(c36):
        l = cnt[None] - c36 * 36 - c81 * 81 - c144 * 144 + i * 36
        r = l + 36
        val[l:r] = make_semi_positive_definite(val[l:r], 6)
    for i in range(c81):
        l = cnt[None] - c81 * 81 - c144 * 144 + i * 81
        r = l + 81
        val[l:r] = make_semi_positive_definite(val[l:r], 9)
    for i in range(c144):
        l = cnt[None] - c144 * 144 + i * 144
        r = l + 144
        val[l:r] = make_semi_positive_definite(val[l:r], 12)

    dirichlet_value.fill(0)
    for i in range(cnt[None]):
        if dirichlet_fixed[col[i]]:
            rhs[row[i]] -= dirichlet_value[col[i]] * val[i]
        if dirichlet_fixed[row[i]] or dirichlet_fixed[col[i]]:
            val[i] = 0
    indices = np.where(dirichlet_fixed)
    for i in indices[0]:
        row = np.append(row, i)
        col = np.append(col, i)
        val = np.append(val, 1.)
        rhs[i] = dirichlet_value[i]

    n = n_particles * dim
    A = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n))
    data_sol.from_numpy(scipy.sparse.linalg.spsolve(A, rhs))


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
            residual = ti.max(residual, ti.abs(data_sol[i * dim + d]))
    print("Search Direction Residual : ", residual / dt)
    return residual


if dim == 2:
    gui = ti.GUI("IPC", (1024, 1024), background_color=0x112F41)
else:
    scene = t3.Scene()
    model = t3.Model(f_n=n_boundary_triangles, vi_n=n_particles)
    scene.add_model(model)
    camera = t3.Camera((768, 768))
    scene.add_camera(camera)
    light = t3.Light([0.4, -1.5, 1.8])
    scene.add_light(light)
    gui = ti.GUI('IPC', camera.res)
def write_image(f):
    find_constraints()
    particle_pos = (x.to_numpy() + mesh_offset) * mesh_scale
    vertices_ = vertices.to_numpy()
    if dim == 2:
        for i in range(n_elements):
            for j in range(3):
                a, b = vertices_[i, j], vertices_[i, (j + 1) % 3]
                gui.line((particle_pos[a][0], particle_pos[a][1]),
                         (particle_pos[b][0], particle_pos[b][1]),
                         radius=1,
                         color=0x4FB99F)
        for i in dirichlet:
            gui.circle(particle_pos[i], radius=3, color=0x44FFFF)
        for i in range(n_PE[None]):
            gui.circle(particle_pos[PE[i, 0]], radius=3, color=0xFF4444)
        gui.show(directory + f'images/{f:06d}.png')
    else:
        model.vi.from_numpy(particle_pos.astype(np.float64))
        model.faces.from_numpy(boundary_triangles_.astype(np.int32))
        camera.from_mouse(gui)
        scene.render()
        gui.set_image(camera.img)
        gui.show(directory + f'images/{f:06d}.png')
        f = open(f'output/{f:06d}.obj', 'w')
        for i in range(n_particles):
            f.write('v %.6f %.6f %.6f\n' % (particle_pos[i, 0], particle_pos[i, 1], particle_pos[i, 2]))
        for [p0, p1, p2] in boundary_triangles_:
            f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
        f.close()


if __name__ == "__main__":
    x.from_numpy(mesh_particles.astype(np.float64))
    v.fill(0)
    vertices.from_numpy(mesh_elements.astype(np.int32))
    boundary_points.from_numpy(np.array(list(boundary_points_)).astype(np.int32))
    boundary_edges.from_numpy(boundary_edges_.astype(np.int32))
    boundary_triangles.from_numpy(boundary_triangles_.astype(np.int32))
    dirichlet.from_numpy(dirichlet_fixed.astype(np.int32))
    compute_restT_and_m()
    save_x0()
    zero.fill(0)
    write_image(0)
    total_time = 0.0
    for f in range(120):
        total_time -= time.time()
        print("==================== Frame: ", f, " ====================")
        compute_xn_and_xTilde()
        move_nodes()
        find_constraints()
        while True:
            data_row.fill(0)
            data_col.fill(0)
            data_val.fill(0)
            data_rhs.fill(0)
            data_sol.fill(0)
            compute_hessian_and_gradient()
            solve_system()
            if output_residual() < 1e-2 * dt:
                break
            E0 = compute_energy()
            save_xPrev()
            alpha = compute_intersection_free_step_size()
            apply_sol(alpha)
            find_constraints()
            E = compute_energy()
            while E > E0:
                alpha *= 0.5
                apply_sol(alpha)
                find_constraints()
                E = compute_energy()
            print("!!!!!!!!!!!!!!!!!!!!!", alpha, E)
        compute_v()
        total_time += time.time()
        print("Time : ", total_time)
        write_image(f + 1)
