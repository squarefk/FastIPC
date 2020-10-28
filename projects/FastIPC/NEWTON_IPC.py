from reader import *
from common.physics.fixed_corotated import *
from common.math.math_tools import *
from common.math.ipc import *
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

mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, gravity, dim = read(int(sys.argv[1]))
boundary_points_ = set()
boundary_edges_ = np.zeros(shape=(0, 2), dtype=np.int32)
boundary_triangles_ = np.zeros(shape=(0, 3), dtype=np.int32)

if int(sys.argv[1]) == 1004:
    for i in range(9):
        p0 = 3200 + i * 3
        p1 = 3200 + i * 3 + 1
        p2 = 3200 + i * 3 + 2
        boundary_points_.update([p0, p1, p2])
        boundary_edges_ = np.vstack((boundary_edges_, [p0, p1]))
        boundary_edges_ = np.vstack((boundary_edges_, [p1, p2]))
        boundary_edges_ = np.vstack((boundary_edges_, [p2, p0]))
        boundary_triangles_ = np.vstack((boundary_triangles_, [p0, p1, p2]))
elif int(sys.argv[1]) == 1005:
    for i in range(400):
        p = 7034 + i
        boundary_points_.update([p])

if dim == 2:
    edges = set()
    for [i, j, k] in mesh_elements:
        edges.add((i, j))
        edges.add((j, k))
        edges.add((k, i))
    for [i, j, k] in mesh_elements:
        if (j, i) not in edges:
            boundary_points_.update([j, i])
            boundary_edges_ = np.vstack((boundary_edges_, [j, i]))
        if (k, j) not in edges:
            boundary_points_.update([k, j])
            boundary_edges_ = np.vstack((boundary_edges_, [k, j]))
        if (i, k) not in edges:
            boundary_points_.update([i, k])
            boundary_edges_ = np.vstack((boundary_edges_, [i, k]))
    boundary_triangles_ = np.vstack((boundary_triangles_, [-1, -1, -1]))
else:
    triangles = set()
    for [p0, p1, p2, p3] in mesh_elements:
        triangles.add((p0, p2, p1))
        triangles.add((p0, p3, p2))
        triangles.add((p0, p1, p3))
        triangles.add((p1, p2, p3))
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

directory = 'output/' + '_'.join(sys.argv[:2]) + '/'
os.makedirs(directory + 'images/', exist_ok=True)
os.makedirs(directory + 'caches/', exist_ok=True)
os.makedirs(directory + 'objs/', exist_ok=True)
print('output directory:', directory)
# sys.stdout = open(directory + 'log.txt', 'w')
# sys.stderr = open(directory + 'err.txt', 'w')

##############################################################################

real = ti.f64
ti.init(arch=ti.cpu, default_fp=real, make_thread_local=False) #, cpu_max_num_threads=1)

scalar = lambda: ti.field(real)
vec = lambda: ti.Vector.field(dim, real)
mat = lambda: ti.Matrix.field(dim, dim, real)

dt = 0.01
E = 1e5
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
vertices = ti.field(ti.i32)
boundary_points = ti.field(ti.i32)
boundary_edges = ti.field(ti.i32)
boundary_triangles = ti.field(ti.i32)
ti.root.dense(ti.k, n_particles).place(x, x0, xPrev, xTilde, xn, v, m)
ti.root.dense(ti.k, n_particles).place(zero)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.ij, (n_elements, dim + 1)).place(vertices)
ti.root.dense(ti.i, n_boundary_points).place(boundary_points)
ti.root.dense(ti.ij, (n_boundary_edges, 2)).place(boundary_edges)
ti.root.dense(ti.ij, (n_boundary_triangles, 3)).place(boundary_triangles)

MAX_LINEAR = 5000000
data_rhs = ti.field(real, shape=n_particles * dim)
data_row = ti.field(ti.i32, shape=MAX_LINEAR)
data_col = ti.field(ti.i32, shape=MAX_LINEAR)
data_val = ti.field(real, shape=MAX_LINEAR)
data_sol = ti.field(real, shape=n_particles * dim)
cnt = ti.field(ti.i32, shape=())

MAX_C = 100000
PP = ti.field(ti.i32, shape=(MAX_C, 2))
n_PP = ti.field(ti.i32, shape=())
PE = ti.field(ti.i32, shape=(MAX_C, 3))
n_PE = ti.field(ti.i32, shape=())
PT = ti.field(ti.i32, shape=(MAX_C, 4))
n_PT = ti.field(ti.i32, shape=())
EE = ti.field(ti.i32, shape=(MAX_C, 4))
n_EE = ti.field(ti.i32, shape=())
EEM = ti.field(ti.i32, shape=(MAX_C, 4))
n_EEM = ti.field(ti.i32, shape=())
PPM = ti.field(ti.i32, shape=(MAX_C, 4))
n_PPM = ti.field(ti.i32, shape=())
PEM = ti.field(ti.i32, shape=(MAX_C, 4))
n_PEM = ti.field(ti.i32, shape=())

dHat2 = 1e-5
dHat = dHat2 ** 0.5
kappa = 1e4

pid = ti.field(ti.i32)
if dim == 2:
    indices = ti.ij
else:
    indices = ti.ijk
grid_size = 4096
offset = tuple(-grid_size // 2 for _ in range(dim))
grid_block_size = 128
grid = ti.root.pointer(indices, grid_size // grid_block_size)
if dim == 2:
    leaf_block_size = 16
else:
    leaf_block_size = 8
block = grid.pointer(indices, grid_block_size // leaf_block_size)
block.dynamic(ti.indices(dim), 1024 * 1024, chunk_size=leaf_block_size**dim * 8).place(pid, offset=offset + (0, ))


@ti.kernel
def compute_adaptive_kappa() -> real:
    H_b = barrier_H(1.0e-16, dHat2, 1)
    total_mass = 0.0
    for i in range(n_particles):
        total_mass += m[i]
    return 1.0e13 * total_mass / n_particles / (4.0e-16 * H_b)


@ti.kernel
def find_constraints_2D_PE():
    for i in boundary_points:
        p = boundary_points[i]
        for j in range(n_boundary_edges):
            e0 = boundary_edges[j, 0]
            e1 = boundary_edges[j, 1]
            if p != e0 and p != e1 and point_edge_ccd_broadphase(x[p], x[e0], x[e1], dHat):
                case = PE_type(x[p], x[e0], x[e1])
                if case == 0:
                    if PP_2D_E(x[p], x[e0]) < dHat2:
                        n = ti.atomic_add(n_PP[None], 1)
                        PP[n, 0], PP[n, 1] = min(p, e0), max(p, e0)
                elif case == 1:
                    if PP_2D_E(x[p], x[e1]) < dHat2:
                        n = ti.atomic_add(n_PP[None], 1)
                        PP[n, 0], PP[n, 1] = min(p, e1), max(p, e1)
                elif case == 2:
                    if PE_2D_E(x[p], x[e0], x[e1]) < dHat2:
                        n = ti.atomic_add(n_PE[None], 1)
                        PE[n, 0], PE[n, 1], PE[n, 2] = p, e0, e1
    #
    # inv_dx = 1 / 0.01
    # for i in range(n_boundary_edges):
    #     e0 = boundary_edges[i, 0]
    #     e1 = boundary_edges[i, 1]
    #     lower = int(ti.floor((ti.min(x[e0], x[e1]) - dHat) * inv_dx)) - ti.Vector(list(offset))
    #     upper = int(ti.floor((ti.max(x[e0], x[e1]) + dHat) * inv_dx)) + 1 - ti.Vector(list(offset))
    #     for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]))):
    #         ti.append(pid.parent(), I, i)
    # for i in range(n_boundary_points):
    #     p = boundary_points[i]
    #     lower = int(ti.floor(x[p] * inv_dx)) - ti.Vector(list(offset))
    #     upper = int(ti.floor(x[p] * inv_dx)) + 1 - ti.Vector(list(offset))
    #     for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]))):
    #         L = ti.length(pid.parent(), I)
    #         for l in range(L):
    #             j = pid[I[0], I[1], l]
    #             e0 = boundary_edges[j, 0]
    #             e1 = boundary_edges[j, 1]
    #             if p != e0 and p != e1 and point_edge_ccd_broadphase(x[p], x[e0], x[e1], dHat):
    #                 case = PE_type(x[p], x[e0], x[e1])
    #                 if case == 0:
    #                     if PP_2D_E(x[p], x[e0]) < dHat2:
    #                         n = ti.atomic_add(n_PP[None], 1)
    #                         PP[n, 0], PP[n, 1] = min(p, e0), max(p, e0)
    #                 elif case == 1:
    #                     if PP_2D_E(x[p], x[e1]) < dHat2:
    #                         n = ti.atomic_add(n_PP[None], 1)
    #                         PP[n, 0], PP[n, 1] = min(p, e1), max(p, e1)
    #                 elif case == 2:
    #                     if PE_2D_E(x[p], x[e0], x[e1]) < dHat2:
    #                         n = ti.atomic_add(n_PE[None], 1)
    #                         PE[n, 0], PE[n, 1], PE[n, 2] = p, e0, e1


@ti.kernel
def find_constraints_3D_PT():
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


@ti.kernel
def find_constraints_3D_EE():
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


def find_constraints():
    with Timer("Find constraints"):
        n_PP[None], n_PE[None], n_PT[None], n_EE[None], n_EEM[None], n_PPM[None], n_PEM[None] = 0, 0, 0, 0, 0, 0, 0
        if dim == 2:
            grid.deactivate_all()
            find_constraints_2D_PE()
        else:
            grid.deactivate_all()
            find_constraints_3D_PT()
            grid.deactivate_all()
            find_constraints_3D_EE()
    with Timer("Remove duplicated"):
        xxs = [PP, PE, PT, EE, EEM, PPM, PEM]
        n_xxs = [n_PP, n_PE, n_PT, n_EE, n_EEM, n_PPM, n_PEM]
        for xx, n_xx in zip(xxs, n_xxs):
            tmp = np.unique(xx.to_numpy()[:n_xx[None], :], axis=0)
            n_xx[None] = len(tmp)
            xx.from_numpy(np.resize(tmp, (MAX_C, xx.shape[1])))


@ti.kernel
def compute_intersection_free_step_size() -> real:
    alpha = 1.0
    if ti.static(dim == 2):
        for i in range(n_boundary_points):
            p = boundary_points[i]
            for j in range(n_boundary_edges):
                e0 = boundary_edges[j, 0]
                e1 = boundary_edges[j, 1]
                if p != e0 and p != e1:
                    dp = ti.Vector([data_sol[p * dim + 0], data_sol[p * dim + 1]])
                    de0 = ti.Vector([data_sol[e0 * dim + 0], data_sol[e0 * dim + 1]])
                    de1 = ti.Vector([data_sol[e1 * dim + 0], data_sol[e1 * dim + 1]])
                    if moving_point_edge_ccd_broadphase(x[p], x[e0], x[e1], dp, de0, de1, dHat):
                        alpha = ti.min(alpha, moving_point_edge_ccd(x[p], x[e0], x[e1], dp, de0, de1, 0.2))
    else:
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
    # for i in range(n_elements):
    #     a, b, c, d = vertices[i, 0], vertices[i, 1], vertices[i, 2], vertices[i, 3]
    #     da = ti.Vector([data_sol[a * dim + 0], data_sol[a * dim + 1], data_sol[a * dim + 2]])
    #     db = ti.Vector([data_sol[b * dim + 0], data_sol[b * dim + 1], data_sol[b * dim + 2]])
    #     dc = ti.Vector([data_sol[c * dim + 0], data_sol[c * dim + 1], data_sol[c * dim + 2]])
    #     dd = ti.Vector([data_sol[d * dim + 0], data_sol[d * dim + 1], data_sol[d * dim + 2]])
    #     alpha = ti.min(alpha, get_smallest_positive_real_cubic_root(x[a], x[b], x[c], x[d], da, db, dc, dd, 0.2))
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
        xTilde(1)[i] += dt * dt * gravity


def move_nodes(f):
    if int(sys.argv[1]) == 1001:
        @ti.kernel
        def add_initial_velocity():
            for i in range(n_particles):
                v(0)[i] = 1 if i < n_particles / 2 else -1
        if f == 0:
            add_initial_velocity()
    elif int(sys.argv[1]) == 1002:
        speed = math.pi * 0.4
        for i in range(n_particles):
            if dirichlet_fixed[i]:
                a, b, c = x(0)[i], x(1)[i], x(2)[i]
                angle = ti.atan2(b, c)
                if a < 0:
                    angle += speed * dt
                else:
                    angle -= speed * dt
                radius = ti.sqrt(b * b + c * c)
                dirichlet_value[i, 0] = a
                dirichlet_value[i, 1] = radius * ti.sin(angle)
                dirichlet_value[i, 2] = radius * ti.cos(angle)
    elif int(sys.argv[1]) == 10:
        speed = 1
        for i in range(954):
            if dirichlet_fixed[i]:
                dirichlet_value[i, 0] += speed * dt
    tmp_fixed = np.stack((dirichlet_fixed,) * dim, axis=-1)
    for i in range(n_particles):
        if dirichlet_fixed[i]:
            for d in range(dim):
                x(d)[i] = dirichlet_value[i, d]
                xTilde(d)[i] = dirichlet_value[i, d]
    return np.where(tmp_fixed.reshape((n_particles * dim)))[0], np.zeros((n_particles * dim))


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
        total_energy += PP_energy(x[PP[r, 0]], x[PP[r, 1]], dHat2, kappa)
    for r in range(n_PE[None]):
        total_energy += PE_energy(x[PE[r, 0]], x[PE[r, 1]], x[PE[r, 2]], dHat2, kappa)
    if ti.static(dim == 3):
        for r in range(n_PT[None]):
            total_energy += PT_energy(x[PT[r, 0]], x[PT[r, 1]], x[PT[r, 2]], x[PT[r, 3]], dHat2, kappa)
        for r in range(n_EE[None]):
            total_energy += EE_energy(x[EE[r, 0]], x[EE[r, 1]], x[EE[r, 2]], x[EE[r, 3]], dHat2, kappa)
        for r in range(n_EEM[None]):
            total_energy += EEM_energy(x[EEM[r, 0]], x[EEM[r, 1]], x[EEM[r, 2]], x[EEM[r, 3]], x0[EEM[r, 0]], x0[EEM[r, 1]], x0[EEM[r, 2]], x0[EEM[r, 3]], dHat2, kappa)
        for r in range(n_PPM[None]):
            total_energy += PPM_energy(x[PPM[r, 0]], x[PPM[r, 1]], x[PPM[r, 2]], x[PPM[r, 3]], x0[PPM[r, 0]], x0[PPM[r, 1]], x0[PPM[r, 2]], x0[PPM[r, 3]], dHat2, kappa)
        for r in range(n_PEM[None]):
            total_energy += PEM_energy(x[PEM[r, 0]], x[PEM[r, 1]], x[PEM[r, 2]], x[PEM[r, 3]], x0[PEM[r, 0]], x0[PEM[r, 1]], x0[PEM[r, 2]], x0[PEM[r, 3]], dHat2, kappa)
    return total_energy


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
        g, H = PP_g_and_H(x[PP[r, 0]], x[PP[r, 1]], dHat2, kappa)
        load_hessian_and_gradient(H, g, ti.Vector([PP[r, 0], PP[r, 1]]), cnt[None] + r * dim * dim * 2 * 2)
    cnt[None] += n_PP[None] * dim * dim * 2 * 2
@ti.kernel
def compute_ipc1():
    for r in range(n_PE[None]):
        g, H = PE_g_and_H(x[PE[r, 0]], x[PE[r, 1]], x[PE[r, 2]], dHat2, kappa)
        load_hessian_and_gradient(H, g, ti.Vector([PE[r, 0], PE[r, 1], PE[r, 2]]), cnt[None] + r * dim * dim * 3 * 3)
    cnt[None] += n_PE[None] * dim * dim * 3 * 3
@ti.kernel
def compute_ipc2():
    for r in range(n_PT[None]):
        g, H = PT_g_and_H(x[PT[r, 0]], x[PT[r, 1]], x[PT[r, 2]], x[PT[r, 3]], dHat2, kappa)
        load_hessian_and_gradient(H, g, ti.Vector([PT[r, 0], PT[r, 1], PT[r, 2], PT[r, 3]]), cnt[None] + r * 144)
    cnt[None] += n_PT[None] * 144
@ti.kernel
def compute_ipc3():
    for r in range(n_EE[None]):
        g, H = EE_g_and_H(x[EE[r, 0]], x[EE[r, 1]], x[EE[r, 2]], x[EE[r, 3]], dHat2, kappa)
        load_hessian_and_gradient(H, g, ti.Vector([EE[r, 0], EE[r, 1], EE[r, 2], EE[r, 3]]), cnt[None] + r * 144)
    cnt[None] += n_EE[None] * 144
@ti.kernel
def compute_ipc4():
    for r in range(n_EEM[None]):
        g, H = EEM_g_and_H(x[EEM[r, 0]], x[EEM[r, 1]], x[EEM[r, 2]], x[EEM[r, 3]], x0[EEM[r, 0]], x0[EEM[r, 1]], x0[EEM[r, 2]], x0[EEM[r, 3]], dHat2, kappa)
        load_hessian_and_gradient(H, g, ti.Vector([EEM[r, 0], EEM[r, 1], EEM[r, 2], EEM[r, 3]]), cnt[None] + r * 144)
    cnt[None] += n_EEM[None] * 144
@ti.kernel
def compute_ipc5():
    for r in range(n_PPM[None]):
        g, H = PPM_g_and_H(x[PPM[r, 0]], x[PPM[r, 1]], x[PPM[r, 2]], x[PPM[r, 3]], x0[PPM[r, 0]], x0[PPM[r, 1]], x0[PPM[r, 2]], x0[PPM[r, 3]], dHat2, kappa)
        load_hessian_and_gradient(H, g, ti.Vector([PPM[r, 0], PPM[r, 1], PPM[r, 2], PPM[r, 3]]), cnt[None] + r * 144)
    cnt[None] += n_PPM[None] * 144
@ti.kernel
def compute_ipc6():
    for r in range(n_PEM[None]):
        g, H = PEM_g_and_H(x[PEM[r, 0]], x[PEM[r, 1]], x[PEM[r, 2]], x[PEM[r, 3]], x0[PEM[r, 0]], x0[PEM[r, 1]], x0[PEM[r, 2]], x0[PEM[r, 3]], dHat2, kappa)
        load_hessian_and_gradient(H, g, ti.Vector([PEM[r, 0], PEM[r, 1], PEM[r, 2], PEM[r, 3]]), cnt[None] + r * 144)
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
    print("ipc1 done.")
    if dim == 3:
        compute_ipc2()
        print("ipc2 done.", end='')
        compute_ipc3()
        print("ipc3 done.", end='')
        compute_ipc4()
        print("ipc4 done.", end='')
        compute_ipc5()
        print("ipc5 done.", end='')
        compute_ipc6()
        print("ipc6 done.")


def solve_system(D, V):
    if cnt[None] >= MAX_LINEAR or max(n_PP[None], n_PE[None], n_PT[None], n_EE[None], n_EEM[None], n_PPM[None], n_PEM[None]) >= MAX_C:
        print("FATAL ERROR: Array Too Small!")
    print("Total entries: ", cnt[None])
    with Timer("Taichi to numpy"):
        row, col, val = data_row.to_numpy()[:cnt[None]], data_col.to_numpy()[:cnt[None]], data_val.to_numpy()[:cnt[None]]
        rhs = data_rhs.to_numpy()
    with Timer("DBC"):
        n = n_particles * dim
        A = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n))
        A = scipy.sparse.lil_matrix(A)
        A[:, D] = 0
        A[D, :] = 0
        A = scipy.sparse.csr_matrix(A)
        A += scipy.sparse.csr_matrix((np.ones(len(D)), (D, D)), shape=(n, n))
        rhs[D] = V[D]
    with Timer("System Solve"):
        factor = cholesky(A)
        data_sol.from_numpy(factor(rhs))


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
    gui = ti.GUI("IPC", (768, 768), background_color=0x112F41)
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
    particle_pos = x.to_numpy() * mesh_scale + mesh_offset
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
        boundary_points.from_numpy(np.array(list(boundary_points_)).astype(np.int32))
        boundary_edges.from_numpy(boundary_edges_.astype(np.int32))
        boundary_triangles.from_numpy(boundary_triangles_.astype(np.int32))
        compute_restT_and_m()
        kappa = compute_adaptive_kappa()
        print("Adaptive kappa:", kappa)
        save_x0()
        zero.fill(0)
        write_image(0)
        f_start = 0
        if len(sys.argv) == 3:
            f_start = int(sys.argv[2])
            [x_, v_, dirichlet_fixed, dirichlet_value] = pickle.load(open(directory + f'caches/{f_start:06d}.p', 'rb'))
            x.from_numpy(x_)
            v.from_numpy(v_)
        for f in range(f_start, 360):
            with Timer("Time Step"):
                print("==================== Frame: ", f, " ====================")
                compute_xn_and_xTilde()
                D, V = move_nodes(f)
                with Timer("Find Constraints"):
                    find_constraints()
                while True:
                    with Timer("Build System"):
                        data_row.fill(0)
                        data_col.fill(0)
                        data_val.fill(0)
                        data_rhs.fill(0)
                        data_sol.fill(0)
                        compute_hessian_and_gradient()
                    with Timer("Solve System"):
                        solve_system(D, V)
                    if output_residual() < 1e-2 * dt:
                        break
                    with Timer("Line Search"):
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
                compute_v()
            with Timer("Visualization"):
                write_image(f + 1)
            pickle.dump([x.to_numpy(), v.to_numpy(), dirichlet_fixed, dirichlet_value], open(directory + f'caches/{f + 1:06d}.p', 'wb'))
            Timer_Print()
