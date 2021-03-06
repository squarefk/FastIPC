from reader import *
from common.physics.fixed_corotated import *
from common.math.math_tools import *
from common.math.ipc import *
from common.utils.timer import *
from common.utils.logger import *

import sys, os, time, math
import taichi as ti
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
boundary_points_, boundary_edges_, boundary_triangles_ = settings['boundary']

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
n_boundary_points = len(boundary_points_)
n_boundary_edges = len(boundary_edges_)
n_boundary_triangles = len(boundary_triangles_)

x, x0, xPrev, xTilde, xn, v, m = vec(), vec(), vec(), vec(), vec(), vec(), scalar()
zero = vec()
la, mu = scalar(), scalar()
restT = mat()
vertices = ti.field(ti.i32)
boundary_points = ti.field(ti.i32)
boundary_edges = ti.field(ti.i32)
boundary_triangles = ti.field(ti.i32)
ti.root.dense(ti.i, n_particles).place(x, x0, xPrev, xTilde, xn, v, m)
ti.root.dense(ti.i, n_particles).place(zero)
ti.root.dense(ti.i, n_elements).place(la, mu)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.ij, (n_elements, dim + 1)).place(vertices)
ti.root.dense(ti.i, n_boundary_points).place(boundary_points)
ti.root.dense(ti.ij, (n_boundary_edges, 2)).place(boundary_edges)
ti.root.dense(ti.ij, (n_boundary_triangles, 3)).place(boundary_triangles)

MAX_LINEAR = 50000000 if dim == 3 else 5000000
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

dfx = ti.field(ti.i32, shape=n_particles * dim)

dHat2 = 1e-6
dHat = dHat2 ** 0.5
kappa = 1e4

spatial_hash_inv_dx = 1.0 / 0.01
pid = ti.field(ti.i32)
if dim == 2:
    indices = ti.ij
else:
    indices = ti.ijk
grid_size = 4096
offset = tuple(-grid_size // 4 for _ in range(dim))
grid_block_size = 128
grid = ti.root.pointer(indices, grid_size // grid_block_size)
if dim == 2:
    leaf_block_size = 16
else:
    leaf_block_size = 8
block = grid.pointer(indices, grid_block_size // leaf_block_size)
block.dense(indices, leaf_block_size).dynamic(ti.indices(dim), 1024 * 1024, chunk_size=leaf_block_size**dim * 8).place(pid, offset=offset + (0, ))
offset = ti.Vector(list(offset))


@ti.kernel
def compute_adaptive_kappa() -> real:
    H_b = barrier_H(1.0e-16, dHat2, 1)
    total_mass = 0.0
    for i in range(n_particles):
        total_mass += m[i]
    result = 1.0e13 * total_mass / n_particles / (4.0e-16 * H_b)
    print("Adaptive kappa:", result)
    return result


@ti.kernel
def compute_mean_of_boundary_edges() -> real:
    total = 0.0
    for i in range(n_boundary_edges):
        total += (x[boundary_edges[i, 0]] - x[boundary_edges[i, 1]]).norm()
    result = total / ti.cast(n_boundary_edges, real)
    print("Mean of boundary edges:", result)
    return result


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


@ti.func
def attempt_PT(p, t0, t1, t2):
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
def find_constraints_3D_PT():
    for i in range(n_boundary_points):
        p = boundary_points[i]
        lower = int(ti.floor((x[p] - dHat) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor((x[p] + dHat) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            ti.append(pid.parent(), I, i)
    for i in range(n_boundary_triangles):
        t0 = boundary_triangles[i, 0]
        t1 = boundary_triangles[i, 1]
        t2 = boundary_triangles[i, 2]
        lower = int(ti.floor(min(x[t0], x[t1], x[t2]) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor(max(x[t0], x[t1], x[t2]) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            L = ti.length(pid.parent(), I)
            for l in range(L):
                j = pid[I[0] + offset[0], I[1] + offset[1], I[2] + offset[2], l]
                p = boundary_points[j]
                attempt_PT(p, t0, t1, t2)


@ti.func
def attempt_EE(a0, a1, b0, b1):
    if a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1 and edge_edge_ccd_broadphase(x[a0], x[a1], x[b0], x[b1], dHat):
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
@ti.kernel
def find_constraints_3D_EE():
    for i in range(n_boundary_edges):
        a0 = boundary_edges[i, 0]
        a1 = boundary_edges[i, 1]
        lower = int(ti.floor(min(x[a0], x[a1]) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor(max(x[a0], x[a1]) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            ti.append(pid.parent(), I, i)
    for i in range(n_boundary_edges):
        a0 = boundary_edges[i, 0]
        a1 = boundary_edges[i, 1]
        lower = int(ti.floor((min(x[a0], x[a1]) - dHat) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor((max(x[a0], x[a1]) + dHat) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            L = ti.length(pid.parent(), I)
            for l in range(L):
                j = pid[I[0] + offset[0], I[1] + offset[1], I[2] + offset[2], l]
                b0 = boundary_edges[j, 0]
                b1 = boundary_edges[j, 1]
                if i < j:
                    attempt_EE(a0, a1, b0, b1)


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
            if n_xx[None] == 0:
                continue
            tmp = np.unique(xx.to_numpy()[:n_xx[None], :], axis=0)
            n_xx[None] = len(tmp)
            xx.from_numpy(np.resize(tmp, (MAX_C, xx.shape[1])))
    print("Found constraints:", n_PP[None], n_PE[None], n_PT[None], n_EE[None], n_EEM[None], n_PPM[None], n_PEM[None])


@ti.kernel
def compute_filter_2D_PE() -> real:
    alpha = 1.0
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
                    ti.atomic_min(alpha, point_edge_ccd(x[p], x[e0], x[e1], dp, de0, de1, 0.2))
    return alpha


@ti.kernel
def compute_filter_3D_PT() -> real:
    alpha = 1.0
    for i in range(n_boundary_points):
        p = boundary_points[i]
        dp = ti.Vector([data_sol[p * dim + 0], data_sol[p * dim + 1], data_sol[p * dim + 2]])
        lower = int(ti.floor((min(x[p], x[p] + dp) - dHat) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor((max(x[p], x[p] + dp) + dHat) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            ti.append(pid.parent(), I, i)
    for i in range(n_boundary_triangles):
        t0 = boundary_triangles[i, 0]
        t1 = boundary_triangles[i, 1]
        t2 = boundary_triangles[i, 2]
        dt0 = ti.Vector([data_sol[t0 * dim + 0], data_sol[t0 * dim + 1], data_sol[t0 * dim + 2]])
        dt1 = ti.Vector([data_sol[t1 * dim + 0], data_sol[t1 * dim + 1], data_sol[t1 * dim + 2]])
        dt2 = ti.Vector([data_sol[t2 * dim + 0], data_sol[t2 * dim + 1], data_sol[t2 * dim + 2]])
        lower = int(ti.floor(min(x[t0], x[t0] + dt0, x[t1], x[t1] + dt1, x[t2], x[t2] + dt2) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor(max(x[t0], x[t0] + dt0, x[t1], x[t1] + dt1, x[t2], x[t2] + dt2) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            L = ti.length(pid.parent(), I)
            for l in range(L):
                j = pid[I[0] + offset[0], I[1] + offset[1], I[2] + offset[2], l]
                p = boundary_points[j]
                dp = ti.Vector([data_sol[p * dim + 0], data_sol[p * dim + 1], data_sol[p * dim + 2]])
                if p != t0 and p != t1 and p != t2:
                    if moving_point_triangle_ccd_broadphase(x[p], x[t0], x[t1], x[t2], dp, dt0, dt1, dt2, dHat):
                        dist2 = PT_dist2(x[p], x[t0], x[t1], x[t2], PT_type(x[p], x[t0], x[t1], x[t2]))
                        ti.atomic_min(alpha, point_triangle_ccd(x[p], x[t0], x[t1], x[t2], dp, dt0, dt1, dt2, 0.2, dist2))
    return alpha


@ti.kernel
def compute_filter_3D_EE() -> real:
    alpha = 1.0
    for i in range(n_boundary_edges):
        a0 = boundary_edges[i, 0]
        a1 = boundary_edges[i, 1]
        da0 = ti.Vector([data_sol[a0 * dim + 0], data_sol[a0 * dim + 1], data_sol[a0 * dim + 2]])
        da1 = ti.Vector([data_sol[a1 * dim + 0], data_sol[a1 * dim + 1], data_sol[a1 * dim + 2]])
        lower = int(ti.floor(min(x[a0], x[a0] + da0, x[a1], x[a1] + da1) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor(max(x[a0], x[a0] + da0, x[a1], x[a1] + da1) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            ti.append(pid.parent(), I, i)
    for i in range(n_boundary_edges):
        a0 = boundary_edges[i, 0]
        a1 = boundary_edges[i, 1]
        da0 = ti.Vector([data_sol[a0 * dim + 0], data_sol[a0 * dim + 1], data_sol[a0 * dim + 2]])
        da1 = ti.Vector([data_sol[a1 * dim + 0], data_sol[a1 * dim + 1], data_sol[a1 * dim + 2]])
        lower = int(ti.floor((min(x[a0], x[a0] + da0, x[a1], x[a1] + da1) - dHat) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor((max(x[a0], x[a0] + da0, x[a1], x[a1] + da1) + dHat) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            L = ti.length(pid.parent(), I)
            for l in range(L):
                j = pid[I[0] + offset[0], I[1] + offset[1], I[2] + offset[2], l]
                b0 = boundary_edges[j, 0]
                b1 = boundary_edges[j, 1]
                db0 = ti.Vector([data_sol[b0 * dim + 0], data_sol[b0 * dim + 1], data_sol[b0 * dim + 2]])
                db1 = ti.Vector([data_sol[b1 * dim + 0], data_sol[b1 * dim + 1], data_sol[b1 * dim + 2]])
                if i < j and a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1:
                    if moving_edge_edge_ccd_broadphase(x[a0], x[a1], x[b0], x[b1], da0, da1, db0, db1, dHat):
                        dist2 = EE_dist2(x[a0], x[a1], x[b0], x[b1], EE_type(x[a0], x[a1], x[b0], x[b1]))
                        ti.atomic_min(alpha, edge_edge_ccd(x[a0], x[a1], x[b0], x[b1], da0, da1, db0, db1, 0.2, dist2))
    return alpha


@ti.kernel
def compute_filter_3D_inversion_free() -> real:
    alpha = 1.0
    for i in range(n_elements):
        a, b, c, d = vertices[i, 0], vertices[i, 1], vertices[i, 2], vertices[i, 3]
        da = ti.Vector([data_sol[a * dim + 0], data_sol[a * dim + 1], data_sol[a * dim + 2]])
        db = ti.Vector([data_sol[b * dim + 0], data_sol[b * dim + 1], data_sol[b * dim + 2]])
        dc = ti.Vector([data_sol[c * dim + 0], data_sol[c * dim + 1], data_sol[c * dim + 2]])
        dd = ti.Vector([data_sol[d * dim + 0], data_sol[d * dim + 1], data_sol[d * dim + 2]])
        ti.atomic_min(alpha, get_smallest_positive_real_cubic_root(x[a], x[b], x[c], x[d], da, db, dc, dd, 0.2))
    return alpha


def compute_intersection_free_step_size():
    alpha = 1.0
    if dim == 2:
        grid.deactivate_all()
        alpha = min(alpha, compute_filter_2D_PE())
    else:
        grid.deactivate_all()
        alpha = min(alpha, compute_filter_3D_PT())
        grid.deactivate_all()
        alpha = min(alpha, compute_filter_3D_EE())
        if 'common.physics.neo_hookean' in sys.modules:
            alpha = min(alpha, compute_filter_3D_inversion_free())
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
        F = compute_T(e) @ restT[e].inverse()
        vol0 = restT[e].determinant() / dim / (dim - 1)
        U, sig, V = ti.svd(F)
        total_energy += elasticity_energy(sig, la[e], mu[e]) * dt * dt * vol0
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
    for e in range(n_elements):
        F = compute_T(e) @ restT[e].inverse()
        IB = restT[e].inverse()
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
    compute_inertia()
    compute_elasticity()
    compute_ipc0()
    compute_ipc1()
    if dim == 3:
        compute_ipc2()
        compute_ipc3()
        compute_ipc4()
        compute_ipc5()
        compute_ipc6()


def solve_system(current_time):
    dirichlet_fixed, dirichlet_value = settings['dirichlet'](current_time)
    D, V = np.stack((dirichlet_fixed,) * dim, axis=-1).reshape((n_particles * dim)), np.zeros((n_particles * dim))
    if cnt[None] >= MAX_LINEAR or max(n_PP[None], n_PE[None], n_PT[None], n_EE[None], n_EEM[None], n_PPM[None], n_PEM[None]) >= MAX_C:
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


@ti.kernel
def output_current_minimal_distance():
    d = 1999999999.0
    if ti.static(dim == 2):
        for r in range(n_PP[None]):
            ti.atomic_min(d, PP_2D_E(x[PP[r, 0]], x[PP[r, 1]]))
        for r in range(n_PE[None]):
            ti.atomic_min(d, PE_2D_E(x[PE[r, 0]], x[PE[r, 1]], x[PE[r, 2]]))
    else:
        for r in range(n_PP[None]):
            ti.atomic_min(d, PP_3D_E(x[PP[r, 0]], x[PP[r, 1]]))
        for r in range(n_PE[None]):
            ti.atomic_min(d, PE_3D_E(x[PE[r, 0]], x[PE[r, 1]], x[PE[r, 2]]))
        for r in range(n_PT[None]):
            ti.atomic_min(d, PT_3D_E(x[PT[r, 0]], x[PT[r, 1]], x[PT[r, 2]], x[PT[r, 3]]))
        for r in range(n_EE[None]):
            ti.atomic_min(d, EE_3D_E(x[EE[r, 0]], x[EE[r, 1]], x[EE[r, 2]], x[EE[r, 3]]))
        for r in range(n_PPM[None]):
            ti.atomic_min(d, PP_3D_E(x[PPM[r, 0]], x[PPM[r, 2]]))
        for r in range(n_PEM[None]):
            ti.atomic_min(d, PE_3D_E(x[PEM[r, 0]], x[PEM[r, 2]], x[PEM[r, 3]]))
        for r in range(n_EEM[None]):
            ti.atomic_min(d, EE_3D_E(x[EEM[r, 0]], x[EEM[r, 1]], x[EEM[r, 2]], x[EEM[r, 3]]))

    print("Current minimal distance square", d)


@ti.kernel
def check_collision_2D() -> ti.i32:
    result = 0
    for i in range(n_boundary_points):
        P = boundary_points[i]
        for j in range(n_elements):
            A = vertices[j, 0]
            B = vertices[j, 1]
            C = vertices[j, 2]
            if P != A and P != B and P != C:
                if point_inside_triangle(x[P], x[A], x[B], x[C]):
                    result = 1
    return result
@ti.kernel
def check_collision_3d() -> ti.i32:
    result = 0
    for i in range(n_boundary_edges):
        a0 = boundary_edges[i, 0]
        a1 = boundary_edges[i, 1]
        lower = int(ti.floor((min(x[a0], x[a1]) - dHat) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor((max(x[a0], x[a1]) + dHat) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            ti.append(pid.parent(), I, i)
    for i in range(n_boundary_triangles):
        t0 = boundary_triangles[i, 0]
        t1 = boundary_triangles[i, 1]
        t2 = boundary_triangles[i, 2]
        lower = int(ti.floor(min(x[t0], x[t1], x[t2]) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor(max(x[t0], x[t1], x[t2]) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            L = ti.length(pid.parent(), I)
            for l in range(L):
                j = pid[I[0] + offset[0], I[1] + offset[1], I[2] + offset[2], l]
                a0 = boundary_edges[j, 0]
                a1 = boundary_edges[j, 1]
                if a0 != t0 and a0 != t1 and a0 != t2 and a1 != t0 and a1 != t1 and a1 != t2:
                    if segment_intersect_triangle(x[a0], x[a1], x[t0], x[t1], x[t2]):
                        result = 1
    return result
def check_collision():
    if dim == 2:
        return check_collision_2D()
    else:
        grid.deactivate_all()
        return check_collision_3d()

if dim == 2:
    gui = ti.GUI("IPC", (768, 768), background_color=0x112F41)
else:
    pass
    # scene = t3.Scene()
    # model = t3.Model(f_n=n_boundary_triangles, vi_n=n_particles)
    # scene.add_model(model)
    # camera = t3.Camera((768, 768))
    # scene.add_camera(camera)
    # light = t3.Light([0.4, -1.5, 1.8])
    # scene.add_light(light)
    # gui = ti.GUI('IPC', camera.res)
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
        # model.vi.from_numpy(particle_pos.astype(np.float32))
        # model.faces.from_numpy(boundary_triangles_.astype(np.int32))
        # camera.from_mouse(gui)
        # scene.render()
        # gui.set_image(camera.img)
        # gui.show(directory + f'images/{f:06d}.png')
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
        kappa = compute_adaptive_kappa() * 10
        spatial_hash_inv_dx = 1.0 / compute_mean_of_boundary_edges()
        save_x0()
        zero.fill(0)
        write_image(0)
        f_start = 0
        if settings['start_frame'] > 0:
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
                        find_constraints()
                        output_current_minimal_distance()
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
                        alpha = compute_intersection_free_step_size()
                        print("[Step size after CCD: ", alpha, "]")
                        apply_sol(alpha)
                        while check_collision():
                            alpha /= 2.0
                            apply_sol(alpha)
                        find_constraints()
                        E = compute_energy()
                        while E > E0:
                            alpha *= 0.5
                            apply_sol(alpha)
                            find_constraints()
                            E = compute_energy()
                        while check_collision():
                            alpha /= 2.0
                            apply_sol(alpha)
                        print("[Step size after line search: ", alpha, "]")
                compute_v()
                newton_iter_total += newton_iter
                print("Avg Newton iter: ", newton_iter_total / (f + 1))
            with Timer("Visualization"):
                write_image(f + 1)
            pickle.dump([x.to_numpy(), v.to_numpy()], open(directory + f'caches/{f + 1:06d}.p', 'wb'))
            Timer_Print()
