from reader import *
from common.math.ipc import *
from common.math.math_tools import *
from common.utils.timer import *
from common.utils.logger import *
from common.physics.fixed_corotated import *

from hashlib import sha1
import sys, os, math
import taichi as ti
import taichi_three as t3
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.sparse
import scipy.sparse.linalg
from sksparse.cholmod import *

##############################################################################
settings = read()
directory = settings['directory']
mesh_particles = settings['mesh_particles']
mesh_elements = settings['mesh_elements']
dim = settings['dim']
gravity = settings['gravity']
boundary_points_, boundary_edges_, boundary_triangles_ = settings['boundary']

##############################################################################
real = ti.f64
ti.init(arch=ti.cpu, default_fp=real) #, cpu_max_num_threads=1)
scalar = lambda: ti.field(real)
vec = lambda: ti.Vector.field(dim, real)
mat = lambda: ti.Matrix.field(dim, dim, real)

dt = settings['dt']
E = settings['E']
nu = 0.4
la = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
density = 1000
n_particles = len(mesh_particles)
n_elements = len(mesh_elements)
n_boundary_points = len(boundary_points_)
n_boundary_edges = len(boundary_edges_)
n_boundary_triangles = len(boundary_triangles_)

x, x0, xx, xTilde, xn, v, m = vec(), vec(), vec(), vec(), vec(), vec(), scalar()
restT = mat()
vertices = ti.field(ti.i32)
W, z, zz, u = scalar(), mat(), mat(), mat()
boundary_points = ti.field(ti.i32)
boundary_edges = ti.field(ti.i32)
boundary_triangles = ti.field(ti.i32)
ti.root.dense(ti.i, n_particles).place(x, x0, xx, xTilde, xn, v, m)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.ij, (n_elements, dim + 1)).place(vertices)
ti.root.dense(ti.i, n_elements).place(W, z, zz, u)
ti.root.dense(ti.i, n_boundary_points).place(boundary_points)
ti.root.dense(ti.ij, (n_boundary_edges, 2)).place(boundary_edges)
ti.root.dense(ti.ij, (n_boundary_triangles, 3)).place(boundary_triangles)

drf = ti.field(ti.i32)
ti.root.dense(ti.i, n_particles).place(drf)
drv = scalar()
ti.root.dense(ti.i, n_particles * dim).place(drv)

MAX_LINEAR = 50000000
data_rhs = ti.field(real, shape=n_particles * dim)
_data_row = np.zeros(MAX_LINEAR, dtype=np.int32)
_data_col = np.zeros(MAX_LINEAR, dtype=np.int32)
_data_val = np.zeros(MAX_LINEAR, dtype=np.float64)
data_x = ti.field(real, shape=n_particles * dim)
cnt = ti.field(ti.i32, shape=())

MAX_C = 100000
PP = ti.field(ti.i32, shape=(MAX_C, 2))
n_PP = ti.field(ti.i32, shape=())
y_PP, r_PP, Q_PP = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 1)).place(y_PP, r_PP, Q_PP)
PE = ti.field(ti.i32, shape=(MAX_C, 3))
n_PE = ti.field(ti.i32, shape=())
y_PE, r_PE, Q_PE = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 2)).place(y_PE, r_PE, Q_PE)
PT = ti.field(ti.i32, shape=(MAX_C, 4))
n_PT = ti.field(ti.i32, shape=())
y_PT, r_PT, Q_PT = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_PT, r_PT, Q_PT)
EE = ti.field(ti.i32, shape=(MAX_C, 4))
n_EE = ti.field(ti.i32, shape=())
y_EE, r_EE, Q_EE = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_EE, r_EE, Q_EE)
EEM = ti.field(ti.i32, shape=(MAX_C, 4))
n_EEM = ti.field(ti.i32, shape=())
y_EEM, r_EEM, Q_EEM = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_EEM, r_EEM, Q_EEM)
PPM = ti.field(ti.i32, shape=(MAX_C, 4))
n_PPM = ti.field(ti.i32, shape=())
y_PPM, r_PPM, Q_PPM = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_PPM, r_PPM, Q_PPM)
PEM = ti.field(ti.i32, shape=(MAX_C, 4))
n_PEM = ti.field(ti.i32, shape=())
y_PEM, r_PEM, Q_PEM = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_PEM, r_PEM, Q_PEM)

old_PP = ti.field(ti.i32, shape=(MAX_C, 2))
old_n_PP = ti.field(ti.i32, shape=())
old_y_PP, old_r_PP, old_Q_PP = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 1)).place(old_y_PP, old_r_PP, old_Q_PP)
old_PE = ti.field(ti.i32, shape=(MAX_C, 3))
old_n_PE = ti.field(ti.i32, shape=())
old_y_PE, old_r_PE, old_Q_PE = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 2)).place(old_y_PE, old_r_PE, old_Q_PE)
old_PT = ti.field(ti.i32, shape=(MAX_C, 4))
old_n_PT = ti.field(ti.i32, shape=())
old_y_PT, old_r_PT, old_Q_PT = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_PT, old_r_PT, old_Q_PT)
old_EE = ti.field(ti.i32, shape=(MAX_C, 4))
old_n_EE = ti.field(ti.i32, shape=())
old_y_EE, old_r_EE, old_Q_EE = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_EE, old_r_EE, old_Q_EE)
old_EEM = ti.field(ti.i32, shape=(MAX_C, 4))
old_n_EEM = ti.field(ti.i32, shape=())
old_y_EEM, old_r_EEM, old_Q_EEM = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_EEM, old_r_EEM, old_Q_EEM)
old_PPM = ti.field(ti.i32, shape=(MAX_C, 4))
old_n_PPM = ti.field(ti.i32, shape=())
old_y_PPM, old_r_PPM, old_Q_PPM = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_PPM, old_r_PPM, old_Q_PPM)
old_PEM = ti.field(ti.i32, shape=(MAX_C, 4))
old_n_PEM = ti.field(ti.i32, shape=())
old_y_PEM, old_r_PEM, old_Q_PEM = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_PEM, old_r_PEM, old_Q_PEM)

dfx = ti.field(ti.i32, shape=n_particles)
dfv = ti.field(real, shape=n_particles * dim)

dHat2 = 1e-5 if dim == 2 else 1e-6
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

update_dbdf = False
update_dpdf = False


@ti.kernel
def compute_adaptive_kappa() -> real:
    H_b = barrier_H(1.0e-16, dHat2, 1)
    total_mass = 0.0
    for i in range(n_particles):
        total_mass += m[i]
    return 1.0e13 * total_mass / n_particles / (4.0e-16 * H_b)


@ti.kernel
def compute_mean_of_boundary_edges() -> real:
    total = 0.0
    for i in range(n_boundary_edges):
        total += (x[boundary_edges[i, 0]] - x[boundary_edges[i, 1]]).norm()
    result = total / ti.cast(n_boundary_edges, real)
    print("Mean of boundary edges:", result)
    return result


@ti.kernel
def compute_filter_2D_PE(xx: ti.template(), x: ti.template()) -> real:
    alpha = 1.0
    for i in range(n_boundary_points):
        p = boundary_points[i]
        for j in range(n_boundary_edges):
            e0 = boundary_edges[j, 0]
            e1 = boundary_edges[j, 1]
            if p != e0 and p != e1:
                dp = x[p] - xx[p]
                de0 = x[e0] - xx[e0]
                de1 = x[e1] - xx[e1]
                if moving_point_edge_ccd_broadphase(xx[p], xx[e0], xx[e1], dp, de0, de1, dHat):
                    alpha = ti.min(alpha, point_edge_ccd(xx[p], xx[e0], xx[e1], dp, de0, de1, 0.2))
    return alpha
@ti.kernel
def compute_filter_3D_PT(x: ti.template(), xTrial: ti.template()) -> real:
    alpha = 1.0
    for i in range(n_boundary_points):
        p = boundary_points[i]
        dp = xTrial[p] - x[p]
        lower = int(ti.floor((min(x[p], x[p] + dp) - dHat) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor((max(x[p], x[p] + dp) + dHat) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            ti.append(pid.parent(), I, i)
    for i in range(n_boundary_triangles):
        t0 = boundary_triangles[i, 0]
        t1 = boundary_triangles[i, 1]
        t2 = boundary_triangles[i, 2]
        dt0 = xTrial[t0] - x[t0]
        dt1 = xTrial[t1] - x[t1]
        dt2 = xTrial[t2] - x[t2]
        lower = int(ti.floor(min(x[t0], x[t0] + dt0, x[t1], x[t1] + dt1, x[t2], x[t2] + dt2) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor(max(x[t0], x[t0] + dt0, x[t1], x[t1] + dt1, x[t2], x[t2] + dt2) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            L = ti.length(pid.parent(), I)
            for l in range(L):
                j = pid[I[0] + offset[0], I[1] + offset[1], I[2] + offset[2], l]
                p = boundary_points[j]
                dp = xTrial[p] - x[p]
                if p != t0 and p != t1 and p != t2:
                    if moving_point_triangle_ccd_broadphase(x[p], x[t0], x[t1], x[t2], dp, dt0, dt1, dt2, dHat):
                        dist2 = PT_dist2(x[p], x[t0], x[t1], x[t2], PT_type(x[p], x[t0], x[t1], x[t2]))
                        alpha = min(alpha, point_triangle_ccd(x[p], x[t0], x[t1], x[t2], dp, dt0, dt1, dt2, 0.2, dist2))
    return alpha
@ti.kernel
def compute_filter_3D_EE(x: ti.template(), xTrial: ti.template()) -> real:
    alpha = 1.0
    for i in range(n_boundary_edges):
        a0 = boundary_edges[i, 0]
        a1 = boundary_edges[i, 1]
        da0 = xTrial[a0] - x[a0]
        da1 = xTrial[a1] - x[a1]
        lower = int(ti.floor(min(x[a0], x[a0] + da0, x[a1], x[a1] + da1) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor(max(x[a0], x[a0] + da0, x[a1], x[a1] + da1) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            ti.append(pid.parent(), I, i)
    for i in range(n_boundary_edges):
        a0 = boundary_edges[i, 0]
        a1 = boundary_edges[i, 1]
        da0 = xTrial[a0] - x[a0]
        da1 = xTrial[a1] - x[a1]
        lower = int(ti.floor((min(x[a0], x[a0] + da0, x[a1], x[a1] + da1) - dHat) * spatial_hash_inv_dx)) - offset
        upper = int(ti.floor((max(x[a0], x[a0] + da0, x[a1], x[a1] + da1) + dHat) * spatial_hash_inv_dx)) + 1 - offset
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            L = ti.length(pid.parent(), I)
            for l in range(L):
                j = pid[I[0] + offset[0], I[1] + offset[1], I[2] + offset[2], l]
                b0 = boundary_edges[j, 0]
                b1 = boundary_edges[j, 1]
                db0 = xTrial[b0] - x[b0]
                db1 = xTrial[b1] - x[b1]
                if i < j and a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1:
                    if moving_edge_edge_ccd_broadphase(x[a0], x[a1], x[b0], x[b1], da0, da1, db0, db1, dHat):
                        dist2 = EE_dist2(x[a0], x[a1], x[b0], x[b1], EE_type(x[a0], x[a1], x[b0], x[b1]))
                        alpha = min(alpha, edge_edge_ccd(x[a0], x[a1], x[b0], x[b1], da0, da1, db0, db1, 0.2, dist2))
    return alpha
def compute_filter(x, xTrial):
    alpha = 1.0
    if dim == 2:
        alpha = min(alpha, compute_filter_2D_PE(x, xTrial))
    else:
        grid.deactivate_all()
        alpha = min(alpha, compute_filter_3D_PT(x, xTrial))
        grid.deactivate_all()
        alpha = min(alpha, compute_filter_3D_EE(x, xTrial))
    return alpha


@ti.func
def compute_T(i, x: ti.template()):
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
    for _ in range(1):
        for i in range(n_elements):
            restT[i] = compute_T(i, x)
            mass = restT[i].determinant() / dim / (dim - 1) * density / (dim + 1)
            if mass < 0.0:
                print("FATAL ERROR : mesh inverted")
            for d in ti.static(range(dim + 1)):
                m[vertices[i, d]] += mass
    for i in range(n_particles):
        x0[i] = x[i]

@ti.kernel
def initial_guess():
    # set W, u, z
    for i in range(n_particles):
        xn[i] = x[i]
        xTilde[i] = x[i] + dt * v[i]
        xTilde(1)[i] += dt * dt * gravity
    for i in range(n_elements):
        currentT = compute_T(i, xTilde)
        vol0 = restT[i].determinant() / dim / (dim - 1)
        F = compute_T(i, x) @ restT[i].inverse()
        U, sig, V = svd(F)
        # W[i] = ti.sqrt(elasticity_hessian(sig, la, mu).norm() * dt * dt * vol0)
        W[i] = ti.sqrt((la + mu * 2 / 3) * dt * dt * vol0)
        z[i] = currentT @ restT[i].inverse()
        u[i] = ti.Matrix.zero(real, dim, dim)
    n_PP[None], n_PE[None], n_PT[None], n_EE[None], n_EEM[None], n_PPM[None], n_PEM[None] = 0, 0, 0, 0, 0, 0, 0


@ti.func
def X2F(p: ti.template(), q: ti.template(), i: ti.template(), j: ti.template(), A):
    val = 0.0
    if ti.static(dim == 2):
        if i == q:
            if p == 1:
                val = A[0, j]
            elif p == 2:
                val = A[1, j]
            elif p == 0:
                val = -A[0, j] - A[1, j]
    else:
        if i == q:
            if p == 1:
                val = A[0, j]
            elif p == 2:
                val = A[1, j]
            elif p == 3:
                val = A[2, j]
            elif p == 0:
                val = -A[0, j] - A[1, j] - A[2, j]
    return val


@ti.kernel
def check_collision_2D() -> ti.i32:
    result = 0
    for i in range(n_boundary_edges):
        P = boundary_edges[i, 0]
        Q = boundary_edges[i, 1]
        for j in range(n_elements):
            A = vertices[j, 0]
            B = vertices[j, 1]
            C = vertices[j, 2]
            if P != A and P != B and P != C and Q != A and Q != B and Q != C:
                if segment_intersect_triangle_2D(x[P], x[Q], x[A], x[B], x[C]):
                    result = 1
    return result
@ti.kernel
def check_collision_3D() -> ti.i32:
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
        return check_collision_3D()


@ti.kernel
def global_step(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
    for i in range(n_particles):
        c = i
        data_row[c] = i
        data_col[c] = i
        data_val[c] = m[i]
        for d in ti.static(range(dim)):
            data_rhs[i * dim + d] += m[i] * xTilde(d)[i]
    cnt[None] += n_particles
    for e in range(n_elements):
        A = restT[e].inverse()
        for p in ti.static(range(dim + 1)):
            for j in ti.static(range(dim)):
                for pp in ti.static(range(dim + 1)):
                    c = cnt[None] + e * (dim + 1) * dim * (dim + 1) + p * dim * (dim + 1) + j * (dim + 1) + pp
                    data_row[c] = vertices[e, p]
                    data_col[c] = vertices[e, pp]
                    data_val[c] = X2F(p, 0, 0, j, A) * X2F(pp, 0, 0, j, A) * W[e] * W[e]
        F = z[e] - u[e]
        for p in ti.static(range(dim + 1)):
            for i in ti.static(range(dim)):
                for j in ti.static(range(dim)):
                    q = i
                    data_rhs[vertices[e, p] * dim + q] += X2F(p, q, i, j, A) * F[i, j] * W[e] * W[e]
    cnt[None] += n_elements * (dim + 1) * dim * (dim + 1)
@ti.kernel
def global_PP(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
    ETE2 = ti.Matrix([[1, -1], [-1, 1]])
    for _ in range(1):
        for c in range(n_PP[None]):
            Q = Q_PP[c, 0]
            for p in ti.static(range(2)):
                for q in ti.static(range(2)):
                    idx = cnt[None] + c * 4 + p * 2 + q
                    data_row[idx] = PP[c, p]
                    data_col[idx] = PP[c, q]
                    data_val[idx] = ETE2[p, q] * Q * Q
            for j in ti.static(range(dim)):
                data_rhs[PP[c, 0] * dim + j] += (y_PP(j)[c, 0] - r_PP(j)[c, 0]) * Q * Q
                data_rhs[PP[c, 1] * dim + j] -= (y_PP(j)[c, 0] - r_PP(j)[c, 0]) * Q * Q
    cnt[None] += n_PP[None] * 4
@ti.kernel
def global_PE(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
    ETE3 = ti.Matrix([[2, -1, -1], [-1, 1, 0], [-1, 0, 1]])
    for _ in range(1):
        for c in range(n_PE[None]):
            Q = Q_PE[c, 0]
            for p in ti.static(range(3)):
                for q in ti.static(range(3)):
                    idx = cnt[None] + c * 9 + p * 3 + q
                    data_row[idx] = PE[c, p]
                    data_col[idx] = PE[c, q]
                    data_val[idx] = ETE3[p, q] * Q * Q
            for j in ti.static(range(dim)):
                data_rhs[PE[c, 0] * dim + j] += (y_PE(j)[c, 0] - r_PE(j)[c, 0]) * Q * Q
                data_rhs[PE[c, 0] * dim + j] += (y_PE(j)[c, 1] - r_PE(j)[c, 1]) * Q * Q
                data_rhs[PE[c, 1] * dim + j] -= (y_PE(j)[c, 0] - r_PE(j)[c, 0]) * Q * Q
                data_rhs[PE[c, 2] * dim + j] -= (y_PE(j)[c, 1] - r_PE(j)[c, 1]) * Q * Q
    cnt[None] += n_PE[None] * 9
@ti.kernel
def global_PT(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for _ in range(1):
        for c in range(n_PT[None]):
            Q = Q_PT[c, 0]
            for p in ti.static(range(4)):
                for q in ti.static(range(4)):
                    idx = cnt[None] + c * 16 + p * 4 + q
                    data_row[idx] = PT[c, p]
                    data_col[idx] = PT[c, q]
                    data_val[idx] = ETE4[p, q] * Q * Q
            for j in ti.static(range(3)):
                data_rhs[PT[c, 0] * 3 + j] += (y_PT(j)[c, 0] - r_PT(j)[c, 0]) * Q * Q
                data_rhs[PT[c, 0] * 3 + j] += (y_PT(j)[c, 1] - r_PT(j)[c, 1]) * Q * Q
                data_rhs[PT[c, 0] * 3 + j] += (y_PT(j)[c, 2] - r_PT(j)[c, 2]) * Q * Q
                data_rhs[PT[c, 1] * 3 + j] -= (y_PT(j)[c, 0] - r_PT(j)[c, 0]) * Q * Q
                data_rhs[PT[c, 2] * 3 + j] -= (y_PT(j)[c, 1] - r_PT(j)[c, 1]) * Q * Q
                data_rhs[PT[c, 3] * 3 + j] -= (y_PT(j)[c, 2] - r_PT(j)[c, 2]) * Q * Q
    cnt[None] += n_PT[None] * 16
@ti.kernel
def global_EE(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for _ in range(1):
        for c in range(n_EE[None]):
            Q = Q_EE[c, 0]
            for p in ti.static(range(4)):
                for q in ti.static(range(4)):
                    idx = cnt[None] + c * 16 + p * 4 + q
                    data_row[idx] = EE[c, p]
                    data_col[idx] = EE[c, q]
                    data_val[idx] = ETE4[p, q] * Q * Q
            for j in ti.static(range(3)):
                data_rhs[EE[c, 0] * 3 + j] += (y_EE(j)[c, 0] - r_EE(j)[c, 0]) * Q * Q
                data_rhs[EE[c, 0] * 3 + j] += (y_EE(j)[c, 1] - r_EE(j)[c, 1]) * Q * Q
                data_rhs[EE[c, 0] * 3 + j] += (y_EE(j)[c, 2] - r_EE(j)[c, 2]) * Q * Q
                data_rhs[EE[c, 1] * 3 + j] -= (y_EE(j)[c, 0] - r_EE(j)[c, 0]) * Q * Q
                data_rhs[EE[c, 2] * 3 + j] -= (y_EE(j)[c, 1] - r_EE(j)[c, 1]) * Q * Q
                data_rhs[EE[c, 3] * 3 + j] -= (y_EE(j)[c, 2] - r_EE(j)[c, 2]) * Q * Q
    cnt[None] += n_EE[None] * 16
@ti.kernel
def global_EEM(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for _ in range(1):
        for c in range(n_EEM[None]):
            Q = Q_EEM[c, 0]
            for p in ti.static(range(4)):
                for q in ti.static(range(4)):
                    idx = cnt[None] + c * 16 + p * 4 + q
                    data_row[idx] = EEM[c, p]
                    data_col[idx] = EEM[c, q]
                    data_val[idx] = ETE4[p, q] * Q * Q
            for j in ti.static(range(3)):
                data_rhs[EEM[c, 0] * 3 + j] += (y_EEM(j)[c, 0] - r_EEM(j)[c, 0]) * Q * Q
                data_rhs[EEM[c, 0] * 3 + j] += (y_EEM(j)[c, 1] - r_EEM(j)[c, 1]) * Q * Q
                data_rhs[EEM[c, 0] * 3 + j] += (y_EEM(j)[c, 2] - r_EEM(j)[c, 2]) * Q * Q
                data_rhs[EEM[c, 1] * 3 + j] -= (y_EEM(j)[c, 0] - r_EEM(j)[c, 0]) * Q * Q
                data_rhs[EEM[c, 2] * 3 + j] -= (y_EEM(j)[c, 1] - r_EEM(j)[c, 1]) * Q * Q
                data_rhs[EEM[c, 3] * 3 + j] -= (y_EEM(j)[c, 2] - r_EEM(j)[c, 2]) * Q * Q
    cnt[None] += n_EEM[None] * 16
@ti.kernel
def global_PPM(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for _ in range(1):
        for c in range(n_PPM[None]):
            Q = Q_PPM[c, 0]
            for p in ti.static(range(4)):
                for q in ti.static(range(4)):
                    idx = cnt[None] + c * 16 + p * 4 + q
                    data_row[idx] = PPM[c, p]
                    data_col[idx] = PPM[c, q]
                    data_val[idx] = ETE4[p, q] * Q * Q
            for j in ti.static(range(3)):
                data_rhs[PPM[c, 0] * 3 + j] += (y_PPM(j)[c, 0] - r_PPM(j)[c, 0]) * Q * Q
                data_rhs[PPM[c, 0] * 3 + j] += (y_PPM(j)[c, 1] - r_PPM(j)[c, 1]) * Q * Q
                data_rhs[PPM[c, 0] * 3 + j] += (y_PPM(j)[c, 2] - r_PPM(j)[c, 2]) * Q * Q
                data_rhs[PPM[c, 1] * 3 + j] -= (y_PPM(j)[c, 0] - r_PPM(j)[c, 0]) * Q * Q
                data_rhs[PPM[c, 2] * 3 + j] -= (y_PPM(j)[c, 1] - r_PPM(j)[c, 1]) * Q * Q
                data_rhs[PPM[c, 3] * 3 + j] -= (y_PPM(j)[c, 2] - r_PPM(j)[c, 2]) * Q * Q
    cnt[None] += n_PPM[None] * 16
@ti.kernel
def global_PEM(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for _ in range(1):
        for c in range(n_PEM[None]):
            Q = Q_PEM[c, 0]
            for p in ti.static(range(4)):
                for q in ti.static(range(4)):
                    idx = cnt[None] + c * 16 + p * 4 + q
                    data_row[idx] = PEM[c, p]
                    data_col[idx] = PEM[c, q]
                    data_val[idx] = ETE4[p, q] * Q * Q
            for j in ti.static(range(3)):
                data_rhs[PEM[c, 0] * 3 + j] += (y_PEM(j)[c, 0] - r_PEM(j)[c, 0]) * Q * Q
                data_rhs[PEM[c, 0] * 3 + j] += (y_PEM(j)[c, 1] - r_PEM(j)[c, 1]) * Q * Q
                data_rhs[PEM[c, 0] * 3 + j] += (y_PEM(j)[c, 2] - r_PEM(j)[c, 2]) * Q * Q
                data_rhs[PEM[c, 1] * 3 + j] -= (y_PEM(j)[c, 0] - r_PEM(j)[c, 0]) * Q * Q
                data_rhs[PEM[c, 2] * 3 + j] -= (y_PEM(j)[c, 1] - r_PEM(j)[c, 1]) * Q * Q
                data_rhs[PEM[c, 3] * 3 + j] -= (y_PEM(j)[c, 2] - r_PEM(j)[c, 2]) * Q * Q
    cnt[None] += n_PEM[None] * 16


def solve_system(current_time):
    with Timer("Filter Global Prepare"):
        @ti.kernel
        def before_solve():
            for i in range(n_particles):
                xx[i] = x[i]
        before_solve()

    with Timer("Init DBC"):
        dirichlet_fixed, dirichlet_value = settings['dirichlet'](current_time)
        D, V = dirichlet_fixed, dirichlet_value.reshape((n_particles * dim))
        dfx.from_numpy(D.astype(np.int32))
        dfv.from_numpy(V.astype(np.float64))

    if cnt[None] >= MAX_LINEAR or n_PP[None] >= MAX_C or n_PE[None] >= MAX_C or n_PT[None] >= MAX_C or n_EE[None] >= MAX_C or n_EEM[None] >= MAX_C or n_PPM[None] >= MAX_C or n_PEM[None] >= MAX_C:
        print("FATAL ERROR: Array Too Small!")

    with Timer("Direct Solve (scipy)"):
        with Timer("DBC"):
            @ti.kernel
            def DBC_set_zeros(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
                for i in range(cnt[None]):
                    r, c = data_row[i], data_col[i]
                    if dfx[r]:
                        data_val[i] = 0
                    if dfx[c]:
                        for d in ti.static(range(dim)):
                            data_rhs[r * dim + d] -= data_val[i] * dfv[c * dim + d]
                        data_val[i] = 0
                for i in dfx:
                    if dfx[i]:
                        idx = ti.atomic_add(cnt[None], 1)
                        data_row[idx] = i
                        data_col[idx] = i
                        data_val[idx] = 1
                        for d in ti.static(range(dim)):
                            data_rhs[i * dim + d] = dfv[i * dim + d]
            DBC_set_zeros(_data_row, _data_col, _data_val)
        with Timer("Taichi_Triplets to Scipy_Triplets"):
            row, col, val = _data_row[:cnt[None]], _data_col[:cnt[None]], _data_val[:cnt[None]]
            rhs = data_rhs.to_numpy()
        with Timer("Scipy_Triplets to Scipy_CSC"):
            A = scipy.sparse.csc_matrix((val, (row, col)), shape=(n_particles, n_particles))
        with Timer("CHOLMOD"):
            factor = cholesky(A)
            residual = 0.
            solved_x = rhs.copy()
            for d in range(dim):
                solved_x[d::dim] = factor(rhs[d::dim])
                residual = max(residual, np.linalg.norm(A.dot(solved_x[d::dim]) - rhs[d::dim], ord=np.inf))
            print("Global solve residual = ", residual)
            data_x.from_numpy(solved_x)
            @ti.kernel
            def after_solve() -> real:
                for i in range(n_particles):
                    for d in ti.static(range(dim)):
                        x(d)[i] = data_x[i * dim + d]
            after_solve()

    with Timer("Filter Global"):
        @ti.kernel
        def op0(alpha: real):
            for i in range(n_particles):
                x[i] = x[i] * alpha + xx[i] * (1 - alpha)
        @ti.kernel
        def op1():
            for i in range(n_particles):
                x[i] = (x[i] + xx[i]) / 2.0
        @ti.kernel
        def op2():
            for i in range(n_particles):
                x[i] = xx[i]
        alpha = compute_filter(xx, x)
        op0(alpha)
        while alpha >= 1e-6 and check_collision() == 1:
            op1()
            alpha /= 2.0
        if alpha < 1e-6:
            print('FATAL ERROR: global filter makes line search step size too small', alpha)
            # exit(0)
            op2()
    return alpha


@ti.kernel
def local_elasticity():
    for e in range(n_elements):
        currentT = compute_T(e, x)
        Dx_plus_u_mtr = currentT @ restT[e].inverse() + u[e]
        U, sig, V = svd(Dx_plus_u_mtr)
        sigma = ti.Matrix.zero(real, dim)
        for i in ti.static(range(dim)):
            sigma[i] = sig[i, i]
        sigma_Dx_plus_u = sigma
        vol0 = restT[e].determinant() / dim / (dim - 1)
        We = W[e]
        converge = False
        iter = 0
        while not converge:
            g = elasticity_gradient(sigma, la, mu) * dt * dt * vol0 + (sigma - sigma_Dx_plus_u) * We * We
            P = project_pd(elasticity_hessian(sigma, la, mu)) * dt * dt * vol0 + ti.Matrix.identity(real, dim) * We * We
            p = -P.inverse() @ g
            iter += 1
            if p.norm_sqr() < 1e-6:
                converge = True
            if iter & 31 == 0:
                print("local elasticity iters:", iter, ", residual:", p.norm_sqr())
            alpha = 1.0
            sigma0 = sigma
            E0 = elasticity_energy(sigma, la, mu) * dt * dt * vol0 + (sigma - sigma_Dx_plus_u).norm_sqr() * We * We / 2
            sigma = sigma0 + p
            E = elasticity_energy(sigma, la, mu) * dt * dt * vol0 + (sigma - sigma_Dx_plus_u).norm_sqr() * We * We / 2
            while E > E0:
                alpha *= 0.5
                sigma = sigma0 + alpha * p
                E = elasticity_energy(sigma, la, mu) * dt * dt * vol0 + (sigma - sigma_Dx_plus_u).norm_sqr() * We * We / 2
        for i in ti.static(range(dim)):
            sig[i, i] = sigma[i]
        z[e] = U @ sig @ V.transpose()
@ti.kernel
def local_PP():
    for c in range(n_PP[None]):
        pos = ti.Matrix.zero(real, dim)
        posTilde = ti.Matrix.zero(real, dim)
        for i in ti.static(range(dim)):
            pos[i] = x(i)[PP[c, 0]] - x(i)[PP[c, 1]]
            posTilde[i] = x(i)[PP[c, 0]] - x(i)[PP[c, 1]] + r_PP(i)[c, 0]
        Q = Q_PP[c, 0]
        op = ti.Matrix.zero(real, dim)
        converge = False
        iter = 0
        while not converge:
            g = PP_gradient(op, extract_vec(pos, list(range(0, dim))), dHat2, kappa) + (pos - posTilde) * Q * Q
            P = PP_hessian(op, extract_vec(pos, list(range(0, dim))), dHat2, kappa) + ti.Matrix.identity(real, dim) * Q * Q
            p = -solve(P, g)
            iter += 1
            if p.norm_sqr() < 1e-6:
                converge = True
            if iter & 31 == 0:
                print("local PP iters:", iter, ", residual:", p.norm_sqr())
            alpha = 1.0
            pos0 = pos
            E0 = PP_energy(op, extract_vec(pos, list(range(0, dim))), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            pos = pos0 + alpha * p
            E = PP_energy(op, extract_vec(pos, list(range(0, dim))), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = PP_energy(op, extract_vec(pos, list(range(0, dim))), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
        for i in ti.static(range(dim)):
            y_PP(i)[c, 0] = pos[i]
@ti.kernel
def local_PE():
    for c in range(n_PE[None]):
        pos = ti.Matrix.zero(real, dim * 2)
        posTilde = ti.Matrix.zero(real, dim * 2)
        for i in ti.static(range(dim)):
            pos[i] = x(i)[PE[c, 0]] - x(i)[PE[c, 1]]
            pos[i + dim] = x(i)[PE[c, 0]] - x(i)[PE[c, 2]]
            posTilde[i] = x(i)[PE[c, 0]] - x(i)[PE[c, 1]] + r_PE(i)[c, 0]
            posTilde[i + dim] = x(i)[PE[c, 0]] - x(i)[PE[c, 2]] + r_PE(i)[c, 1]
        Q = Q_PE[c, 0]
        op = ti.Matrix.zero(real, dim)
        converge = False
        iter = 0
        while not converge:
            g = PE_gradient(op, extract_vec(pos, list(range(0, dim))), extract_vec(pos, list(range(dim, dim * 2))), dHat2, kappa) + (pos - posTilde) * Q * Q
            P = PE_hessian(op, extract_vec(pos, list(range(0, dim))), extract_vec(pos, list(range(dim, dim * 2))), dHat2, kappa) + ti.Matrix.identity(real, dim * 2) * Q * Q
            p = -solve(P, g)
            iter += 1
            if p.norm_sqr() < 1e-6:
                converge = True
            if iter & 31 == 0:
                print("local PE iters:", iter, ", residual:", p.norm_sqr())
            alpha = 1.0
            if ti.static(dim == 2):
                alpha = point_edge_ccd(ti.Vector([0.0, 0.0]), ti.Vector([pos[0], pos[1]]), ti.Vector([pos[2], pos[3]]), ti.Vector([0.0, 0.0]), ti.Vector([p[0], p[1]]), ti.Vector([p[2], p[3]]), 0.1)
            pos0 = pos
            E0 = PE_energy(op, extract_vec(pos, list(range(0, dim))), extract_vec(pos, list(range(dim, dim * 2))), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            pos = pos0 + alpha * p
            E = PE_energy(op, extract_vec(pos, list(range(0, dim))), extract_vec(pos, list(range(dim, dim * 2))), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = PE_energy(op, extract_vec(pos, list(range(0, dim))), extract_vec(pos, list(range(dim, dim * 2))), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
        for i in ti.static(range(dim)):
            y_PE(i)[c, 0] = pos[i]
            y_PE(i)[c, 1] = pos[i + dim]
@ti.kernel
def local_PT():
    for c in range(n_PT[None]):
        pos = ti.Vector([x(0)[PT[c, 0]] - x(0)[PT[c, 1]], x(1)[PT[c, 0]] - x(1)[PT[c, 1]], x(2)[PT[c, 0]] - x(2)[PT[c, 1]],
                         x(0)[PT[c, 0]] - x(0)[PT[c, 2]], x(1)[PT[c, 0]] - x(1)[PT[c, 2]], x(2)[PT[c, 0]] - x(2)[PT[c, 2]],
                         x(0)[PT[c, 0]] - x(0)[PT[c, 3]], x(1)[PT[c, 0]] - x(1)[PT[c, 3]], x(2)[PT[c, 0]] - x(2)[PT[c, 3]]])
        posTilde = ti.Vector([x(0)[PT[c, 0]] - x(0)[PT[c, 1]] + r_PT(0)[c, 0], x(1)[PT[c, 0]] - x(1)[PT[c, 1]] + r_PT(1)[c, 0], x(2)[PT[c, 0]] - x(2)[PT[c, 1]] + r_PT(2)[c, 0],
                              x(0)[PT[c, 0]] - x(0)[PT[c, 2]] + r_PT(0)[c, 1], x(1)[PT[c, 0]] - x(1)[PT[c, 2]] + r_PT(1)[c, 1], x(2)[PT[c, 0]] - x(2)[PT[c, 2]] + r_PT(2)[c, 1],
                              x(0)[PT[c, 0]] - x(0)[PT[c, 3]] + r_PT(0)[c, 2], x(1)[PT[c, 0]] - x(1)[PT[c, 3]] + r_PT(1)[c, 2], x(2)[PT[c, 0]] - x(2)[PT[c, 3]] + r_PT(2)[c, 2]])
        Q = Q_PT[c, 0]
        op = ti.Matrix.zero(real, dim)
        converge = False
        iter = 0
        while not converge:
            g = PT_gradient(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde) * Q * Q
            P = PT_hessian(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + ti.Matrix.identity(real, dim * 3) * Q * Q
            p = -solve(P, g)
            iter += 1
            if p.norm_sqr() < 1e-6:
                converge = True
            if iter & 31 == 0:
                print("local PT iters:", iter, ", residual:", p.norm_sqr())
            alpha = 1.0
            pos0 = pos
            E0 = PT_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            pos = pos0 + alpha * p
            E = PT_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = PT_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
        y_PT[c, 0], y_PT[c, 1], y_PT[c, 2] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
@ti.kernel
def local_EE():
    for c in range(n_EE[None]):
        pos = ti.Vector([x(0)[EE[c, 0]] - x(0)[EE[c, 1]], x(1)[EE[c, 0]] - x(1)[EE[c, 1]], x(2)[EE[c, 0]] - x(2)[EE[c, 1]],
                         x(0)[EE[c, 0]] - x(0)[EE[c, 2]], x(1)[EE[c, 0]] - x(1)[EE[c, 2]], x(2)[EE[c, 0]] - x(2)[EE[c, 2]],
                         x(0)[EE[c, 0]] - x(0)[EE[c, 3]], x(1)[EE[c, 0]] - x(1)[EE[c, 3]], x(2)[EE[c, 0]] - x(2)[EE[c, 3]]])
        posTilde = ti.Vector([x(0)[EE[c, 0]] - x(0)[EE[c, 1]] + r_EE(0)[c, 0], x(1)[EE[c, 0]] - x(1)[EE[c, 1]] + r_EE(1)[c, 0], x(2)[EE[c, 0]] - x(2)[EE[c, 1]] + r_EE(2)[c, 0],
                              x(0)[EE[c, 0]] - x(0)[EE[c, 2]] + r_EE(0)[c, 1], x(1)[EE[c, 0]] - x(1)[EE[c, 2]] + r_EE(1)[c, 1], x(2)[EE[c, 0]] - x(2)[EE[c, 2]] + r_EE(2)[c, 1],
                              x(0)[EE[c, 0]] - x(0)[EE[c, 3]] + r_EE(0)[c, 2], x(1)[EE[c, 0]] - x(1)[EE[c, 3]] + r_EE(1)[c, 2], x(2)[EE[c, 0]] - x(2)[EE[c, 3]] + r_EE(2)[c, 2]])
        Q = Q_EE[c, 0]
        op = ti.Matrix.zero(real, dim)
        converge = False
        iter = 0
        while not converge:
            g = EE_gradient(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde) * Q * Q
            P = EE_hessian(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + ti.Matrix.identity(real, dim * 3) * Q * Q
            p = -solve(P, g)
            iter += 1
            if p.norm_sqr() < 1e-6:
                converge = True
            if iter & 31 == 0:
                print("local EE iters:", iter, ", residual:", p.norm_sqr())
            alpha = 1.0
            pos0 = pos
            E0 = EE_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            pos = pos0 + alpha * p
            E = EE_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = EE_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
        y_EE[c, 0], y_EE[c, 1], y_EE[c, 2] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
@ti.kernel
def local_EEM():
    for c in range(n_EEM[None]):
        pos = ti.Vector([x(0)[EEM[c, 0]] - x(0)[EEM[c, 1]], x(1)[EEM[c, 0]] - x(1)[EEM[c, 1]], x(2)[EEM[c, 0]] - x(2)[EEM[c, 1]],
                         x(0)[EEM[c, 0]] - x(0)[EEM[c, 2]], x(1)[EEM[c, 0]] - x(1)[EEM[c, 2]], x(2)[EEM[c, 0]] - x(2)[EEM[c, 2]],
                         x(0)[EEM[c, 0]] - x(0)[EEM[c, 3]], x(1)[EEM[c, 0]] - x(1)[EEM[c, 3]], x(2)[EEM[c, 0]] - x(2)[EEM[c, 3]]])
        posTilde = ti.Vector([x(0)[EEM[c, 0]] - x(0)[EEM[c, 1]] + r_EEM(0)[c, 0], x(1)[EEM[c, 0]] - x(1)[EEM[c, 1]] + r_EEM(1)[c, 0], x(2)[EEM[c, 0]] - x(2)[EEM[c, 1]] + r_EEM(2)[c, 0],
                              x(0)[EEM[c, 0]] - x(0)[EEM[c, 2]] + r_EEM(0)[c, 1], x(1)[EEM[c, 0]] - x(1)[EEM[c, 2]] + r_EEM(1)[c, 1], x(2)[EEM[c, 0]] - x(2)[EEM[c, 2]] + r_EEM(2)[c, 1],
                              x(0)[EEM[c, 0]] - x(0)[EEM[c, 3]] + r_EEM(0)[c, 2], x(1)[EEM[c, 0]] - x(1)[EEM[c, 3]] + r_EEM(1)[c, 2], x(2)[EEM[c, 0]] - x(2)[EEM[c, 3]] + r_EEM(2)[c, 2]])
        Q = Q_EEM[c, 0]
        op = ti.Matrix.zero(real, dim)
        _a0, _a1, _b0, _b1 = x0[EEM[c, 0]], x0[EEM[c, 1]], x0[EEM[c, 2]], x0[EEM[c, 3]]
        converge = False
        iter = 0
        while not converge:
            g = EEM_gradient(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde) * Q * Q
            P = EEM_hessian(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + ti.Matrix.identity(real, dim * 3) * Q * Q
            p = -solve(P, g)
            iter += 1
            if p.norm_sqr() < 1e-6:
                converge = True
            if iter & 31 == 0:
                print("local EEM iters:", iter, ", residual:", p.norm_sqr())
            alpha = 1.0
            pos0 = pos
            E0 = EEM_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            pos = pos0 + alpha * p
            E = EEM_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = EEM_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
        y_EEM[c, 0], y_EEM[c, 1], y_EEM[c, 2] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
@ti.kernel
def local_PPM():
    for c in range(n_PPM[None]):
        pos = ti.Vector([x(0)[PPM[c, 0]] - x(0)[PPM[c, 1]], x(1)[PPM[c, 0]] - x(1)[PPM[c, 1]], x(2)[PPM[c, 0]] - x(2)[PPM[c, 1]],
                         x(0)[PPM[c, 0]] - x(0)[PPM[c, 2]], x(1)[PPM[c, 0]] - x(1)[PPM[c, 2]], x(2)[PPM[c, 0]] - x(2)[PPM[c, 2]],
                         x(0)[PPM[c, 0]] - x(0)[PPM[c, 3]], x(1)[PPM[c, 0]] - x(1)[PPM[c, 3]], x(2)[PPM[c, 0]] - x(2)[PPM[c, 3]]])
        posTilde = ti.Vector([x(0)[PPM[c, 0]] - x(0)[PPM[c, 1]] + r_PPM(0)[c, 0], x(1)[PPM[c, 0]] - x(1)[PPM[c, 1]] + r_PPM(1)[c, 0], x(2)[PPM[c, 0]] - x(2)[PPM[c, 1]] + r_PPM(2)[c, 0],
                              x(0)[PPM[c, 0]] - x(0)[PPM[c, 2]] + r_PPM(0)[c, 1], x(1)[PPM[c, 0]] - x(1)[PPM[c, 2]] + r_PPM(1)[c, 1], x(2)[PPM[c, 0]] - x(2)[PPM[c, 2]] + r_PPM(2)[c, 1],
                              x(0)[PPM[c, 0]] - x(0)[PPM[c, 3]] + r_PPM(0)[c, 2], x(1)[PPM[c, 0]] - x(1)[PPM[c, 3]] + r_PPM(1)[c, 2], x(2)[PPM[c, 0]] - x(2)[PPM[c, 3]] + r_PPM(2)[c, 2]])
        Q = Q_PPM[c, 0]
        op = ti.Matrix.zero(real, dim)
        _a0, _a1, _b0, _b1 = x0[PPM[c, 0]], x0[PPM[c, 1]], x0[PPM[c, 2]], x0[PPM[c, 3]]
        converge = False
        iter = 0
        while not converge:
            g = PPM_gradient(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde) * Q * Q
            P = PPM_hessian(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + ti.Matrix.identity(real, dim * 3) * Q * Q
            p = -solve(P, g)
            iter += 1
            if p.norm_sqr() < 1e-6:
                converge = True
            if iter & 31 == 0:
                print("local PPM iters:", iter, ", residual:", p.norm_sqr())
            alpha = 1.0
            pos0 = pos
            E0 = PPM_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            pos = pos0 + alpha * p
            E = PPM_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = PPM_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
        y_PPM[c, 0], y_PPM[c, 1], y_PPM[c, 2] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
@ti.kernel
def local_PEM():
    for c in range(n_PEM[None]):
        pos = ti.Vector([x(0)[PEM[c, 0]] - x(0)[PEM[c, 1]], x(1)[PEM[c, 0]] - x(1)[PEM[c, 1]], x(2)[PEM[c, 0]] - x(2)[PEM[c, 1]],
                         x(0)[PEM[c, 0]] - x(0)[PEM[c, 2]], x(1)[PEM[c, 0]] - x(1)[PEM[c, 2]], x(2)[PEM[c, 0]] - x(2)[PEM[c, 2]],
                         x(0)[PEM[c, 0]] - x(0)[PEM[c, 3]], x(1)[PEM[c, 0]] - x(1)[PEM[c, 3]], x(2)[PEM[c, 0]] - x(2)[PEM[c, 3]]])
        posTilde = ti.Vector([x(0)[PEM[c, 0]] - x(0)[PEM[c, 1]] + r_PEM(0)[c, 0], x(1)[PEM[c, 0]] - x(1)[PEM[c, 1]] + r_PEM(1)[c, 0], x(2)[PEM[c, 0]] - x(2)[PEM[c, 1]] + r_PEM(2)[c, 0],
                              x(0)[PEM[c, 0]] - x(0)[PEM[c, 2]] + r_PEM(0)[c, 1], x(1)[PEM[c, 0]] - x(1)[PEM[c, 2]] + r_PEM(1)[c, 1], x(2)[PEM[c, 0]] - x(2)[PEM[c, 2]] + r_PEM(2)[c, 1],
                              x(0)[PEM[c, 0]] - x(0)[PEM[c, 3]] + r_PEM(0)[c, 2], x(1)[PEM[c, 0]] - x(1)[PEM[c, 3]] + r_PEM(1)[c, 2], x(2)[PEM[c, 0]] - x(2)[PEM[c, 3]] + r_PEM(2)[c, 2]])
        Q = Q_PEM[c, 0]
        op = ti.Matrix.zero(real, dim)
        _a0, _a1, _b0, _b1 = x0[PEM[c, 0]], x0[PEM[c, 1]], x0[PEM[c, 2]], x0[PEM[c, 3]]
        converge = False
        iter = 0
        while not converge:
            g = PEM_gradient(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde) * Q * Q
            P = PEM_hessian(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + ti.Matrix.identity(real, dim * 3) * Q * Q
            p = -solve(P, g)
            iter += 1
            if p.norm_sqr() < 1e-6:
                converge = True
            if iter & 31 == 0:
                print("local PEM iters:", iter, ", residual:", p.norm_sqr())
            alpha = 1.0
            pos0 = pos
            E0 = PEM_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            pos = pos0 + alpha * p
            E = PEM_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = PEM_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
        y_PEM[c, 0], y_PEM[c, 1], y_PEM[c, 2] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])


# @ti.kernel
# def prime_residual() -> real:
#     residual = 0.0
#     for i in range(n_elements):
#         currentT = compute_T(i)
#         F = currentT @ restT[i].inverse()
#         residual += (F - z[i]).norm_sqr() * W[i] * W[i]
#     for c in range(cc[None]):
#         residual += (x[constraints[c, 0]] - x[constraints[c, 1]] - y[c, 0]).norm_sqr() * Q[c] * Q[c]
#         residual += (x[constraints[c, 0]] - x[constraints[c, 2]] - y[c, 1]).norm_sqr() * Q[c] * Q[c]
#     return residual
#
#
# @ti.kernel
# def dual_residual() -> real:
#     residual = 0.0
#     for i in data_rhs:
#         data_rhs[i] = 0
#     for e in range(n_elements):
#         A = restT[e].inverse()
#         delta = z[e] - zz[e]
#         for p in ti.static(range(3)):
#             for i in ti.static(range(2)):
#                 for j in ti.static(range(2)):
#                     q = i
#                     data_rhs[vertices[e, p] * 2 + q] += X2F(p, q, i, j, A) * delta[i, j] * W[e] * W[e]
#         zz[e] = z[e]
#     for i in data_rhs:
#         residual += data_rhs[i] * data_rhs[i]
#
#     for i in data_rhs:
#         data_rhs[i] = 0
#
#     for c in range(old_cc[None]):
#         for j in ti.static(range(2)):
#             data_rhs[constraints[c, 0] * 2 + j] += (- old_y(j)[c, 0]) * old_Q[c] * old_Q[c]
#             data_rhs[constraints[c, 0] * 2 + j] += (- old_y(j)[c, 1]) * old_Q[c] * old_Q[c]
#             data_rhs[constraints[c, 1] * 2 + j] -= (- old_y(j)[c, 0]) * old_Q[c] * old_Q[c]
#             data_rhs[constraints[c, 2] * 2 + j] -= (- old_y(j)[c, 1]) * old_Q[c] * old_Q[c]
#     for d in range(cc[None]):
#         for j in ti.static(range(2)):
#             data_rhs[constraints[d, 0] * 2 + j] += (y(j)[d, 0]) * Q[d] * Q[d]
#             data_rhs[constraints[d, 0] * 2 + j] += (y(j)[d, 1]) * Q[d] * Q[d]
#             data_rhs[constraints[d, 1] * 2 + j] -= (y(j)[d, 0]) * Q[d] * Q[d]
#             data_rhs[constraints[d, 2] * 2 + j] -= (y(j)[d, 1]) * Q[d] * Q[d]
#     for i in data_rhs:
#         residual += data_rhs[i] * data_rhs[i]
#     return residual
#
#
# @ti.kernel
# def X_residual() -> real:
#     residual = 0.0
#     for _ in range(1):
#         for i in range(n_particles):
#             residual = max(residual, (xx[i] - x[i]).norm_sqr())
#             xx[i] = x[i]
#     return residual


@ti.kernel
def dual_step():
    for i in range(n_elements):
        currentT = compute_T(i, x)
        F = currentT @ restT[i].inverse()
        u[i] += F - z[i]
    for c in range(n_PP[None]):
        r_PP[c, 0] += x[PP[c, 0]] - x[PP[c, 1]] - y_PP[c, 0]
    for c in range(n_PE[None]):
        r_PE[c, 0] += x[PE[c, 0]] - x[PE[c, 1]] - y_PE[c, 0]
        r_PE[c, 1] += x[PE[c, 0]] - x[PE[c, 2]] - y_PE[c, 1]
    for c in range(n_PT[None]):
        r_PT[c, 0] += x[PT[c, 0]] - x[PT[c, 1]] - y_PT[c, 0]
        r_PT[c, 1] += x[PT[c, 0]] - x[PT[c, 2]] - y_PT[c, 1]
        r_PT[c, 2] += x[PT[c, 0]] - x[PT[c, 3]] - y_PT[c, 2]
    for c in range(n_EE[None]):
        r_EE[c, 0] += x[EE[c, 0]] - x[EE[c, 1]] - y_EE[c, 0]
        r_EE[c, 1] += x[EE[c, 0]] - x[EE[c, 2]] - y_EE[c, 1]
        r_EE[c, 2] += x[EE[c, 0]] - x[EE[c, 3]] - y_EE[c, 2]
    for c in range(n_EEM[None]):
        r_EEM[c, 0] += x[EEM[c, 0]] - x[EEM[c, 1]] - y_EEM[c, 0]
        r_EEM[c, 1] += x[EEM[c, 0]] - x[EEM[c, 2]] - y_EEM[c, 1]
        r_EEM[c, 2] += x[EEM[c, 0]] - x[EEM[c, 3]] - y_EEM[c, 2]
    for c in range(n_PPM[None]):
        r_PPM[c, 0] += x[PPM[c, 0]] - x[PPM[c, 1]] - y_PPM[c, 0]
        r_PPM[c, 1] += x[PPM[c, 0]] - x[PPM[c, 2]] - y_PPM[c, 1]
        r_PPM[c, 2] += x[PPM[c, 0]] - x[PPM[c, 3]] - y_PPM[c, 2]
    for c in range(n_PEM[None]):
        r_PEM[c, 0] += x[PEM[c, 0]] - x[PEM[c, 1]] - y_PEM[c, 0]
        r_PEM[c, 1] += x[PEM[c, 0]] - x[PEM[c, 2]] - y_PEM[c, 1]
        r_PEM[c, 2] += x[PEM[c, 0]] - x[PEM[c, 3]] - y_PEM[c, 2]


@ti.kernel
def backup_admm_variables():
    old_n_PP[None] = n_PP[None]
    for c in range(old_n_PP[None]):
        old_PP[c, 0], old_PP[c, 1] = PP[c, 0], PP[c, 1]
        old_y_PP[c, 0] = y_PP[c, 0]
        old_r_PP[c, 0] = r_PP[c, 0]
        old_Q_PP[c, 0] = Q_PP[c, 0]
    old_n_PE[None] = n_PE[None]
    for c in range(old_n_PE[None]):
        old_PE[c, 0], old_PE[c, 1], old_PE[c, 2] = PE[c, 0], PE[c, 1], PE[c, 2]
        old_y_PE[c, 0], old_y_PE[c, 1] = y_PE[c, 0], y_PE[c, 1]
        old_r_PE[c, 0], old_r_PE[c, 1] = r_PE[c, 0], r_PE[c, 1]
        old_Q_PE[c, 0], old_Q_PE[c, 1] = Q_PE[c, 0], Q_PE[c, 1]
    old_n_PT[None] = n_PT[None]
    for c in range(old_n_PT[None]):
        old_PT[c, 0], old_PT[c, 1], old_PT[c, 2], old_PT[c, 3] = PT[c, 0], PT[c, 1], PT[c, 2], PT[c, 3]
        old_y_PT[c, 0], old_y_PT[c, 1], old_y_PT[c, 2] = y_PT[c, 0], y_PT[c, 1], y_PT[c, 2]
        old_r_PT[c, 0], old_r_PT[c, 1], old_r_PT[c, 2] = r_PT[c, 0], r_PT[c, 1], r_PT[c, 2]
        old_Q_PT[c, 0], old_Q_PT[c, 1], old_Q_PT[c, 2] = Q_PT[c, 0], Q_PT[c, 1], Q_PT[c, 2]
    old_n_EE[None] = n_EE[None]
    for c in range(old_n_EE[None]):
        old_EE[c, 0], old_EE[c, 1], old_EE[c, 2], old_EE[c, 3] = EE[c, 0], EE[c, 1], EE[c, 2], EE[c, 3]
        old_y_EE[c, 0], old_y_EE[c, 1], old_y_EE[c, 2] = y_EE[c, 0], y_EE[c, 1], y_EE[c, 2]
        old_r_EE[c, 0], old_r_EE[c, 1], old_r_EE[c, 2] = r_EE[c, 0], r_EE[c, 1], r_EE[c, 2]
        old_Q_EE[c, 0], old_Q_EE[c, 1], old_Q_EE[c, 2] = Q_EE[c, 0], Q_EE[c, 1], Q_EE[c, 2]
    old_n_EEM[None] = n_EEM[None]
    for c in range(old_n_EEM[None]):
        old_EEM[c, 0], old_EEM[c, 1], old_EEM[c, 2], old_EEM[c, 3] = EEM[c, 0], EEM[c, 1], EEM[c, 2], EEM[c, 3]
        old_y_EEM[c, 0], old_y_EEM[c, 1], old_y_EEM[c, 2] = y_EEM[c, 0], y_EEM[c, 1], y_EEM[c, 2]
        old_r_EEM[c, 0], old_r_EEM[c, 1], old_r_EEM[c, 2] = r_EEM[c, 0], r_EEM[c, 1], r_EEM[c, 2]
        old_Q_EEM[c, 0], old_Q_EEM[c, 1], old_Q_EEM[c, 2] = Q_EEM[c, 0], Q_EEM[c, 1], Q_EEM[c, 2]
    old_n_PPM[None] = n_PPM[None]
    for c in range(old_n_PPM[None]):
        old_PPM[c, 0], old_PPM[c, 1], old_PPM[c, 2], old_PPM[c, 3] = PPM[c, 0], PPM[c, 1], PPM[c, 2], PPM[c, 3]
        old_y_PPM[c, 0], old_y_PPM[c, 1], old_y_PPM[c, 2] = y_PPM[c, 0], y_PPM[c, 1], y_PPM[c, 2]
        old_r_PPM[c, 0], old_r_PPM[c, 1], old_r_PPM[c, 2] = r_PPM[c, 0], r_PPM[c, 1], r_PPM[c, 2]
        old_Q_PPM[c, 0], old_Q_PPM[c, 1], old_Q_PPM[c, 2] = Q_PPM[c, 0], Q_PPM[c, 1], Q_PPM[c, 2]
    old_n_PEM[None] = n_PEM[None]
    for c in range(old_n_PEM[None]):
        old_PEM[c, 0], old_PEM[c, 1], old_PEM[c, 2], old_PEM[c, 3] = PEM[c, 0], PEM[c, 1], PEM[c, 2], PEM[c, 3]
        old_y_PEM[c, 0], old_y_PEM[c, 1], old_y_PEM[c, 2] = y_PEM[c, 0], y_PEM[c, 1], y_PEM[c, 2]
        old_r_PEM[c, 0], old_r_PEM[c, 1], old_r_PEM[c, 2] = r_PEM[c, 0], r_PEM[c, 1], r_PEM[c, 2]
        old_Q_PEM[c, 0], old_Q_PEM[c, 1], old_Q_PEM[c, 2] = Q_PEM[c, 0], Q_PEM[c, 1], Q_PEM[c, 2]
    n_PP[None], n_PE[None], n_PT[None], n_EE[None], n_EEM[None], n_PPM[None], n_PEM[None] = 0, 0, 0, 0, 0, 0, 0


@ti.kernel
def find_constraints_2D_PE():
    for i in range(n_boundary_points):
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


def remove_duplicated_constraints():
    tmp = np.unique(PP.to_numpy()[:n_PP[None], :], axis=0)
    n_PP[None] = len(tmp)
    PP.from_numpy(np.resize(tmp, (MAX_C, 2)))
    tmp = np.unique(PE.to_numpy()[:n_PE[None], :], axis=0)
    n_PE[None] = len(tmp)
    PE.from_numpy(np.resize(tmp, (MAX_C, 3)))
    print("Find constraints: ", n_PP[None], n_PE[None], n_PT[None], n_EE[None], n_EEM[None], n_PPM[None], n_PEM[None])


@ti.kernel
def reuse_admm_variables(alpha: real):
    # xTilde initiated y, r
    min_Q = ti.sqrt(PP_hessian(ti.Matrix.zero(real, dim), ti.Matrix.one(real, dim) * 9e-1 * dHat, dHat2, kappa).norm()) / 10
    max_Q = ti.sqrt(PP_hessian(ti.Matrix.zero(real, dim), ti.Matrix.one(real, dim) * 1e-4 * dHat, dHat2, kappa).norm()) * 10
    ############################################### PP ###############################################
    for r in range(n_PP[None]):
        p0 = xTilde[PP[r, 0]] * alpha + x[PP[r, 0]] * (1 - alpha)
        p1 = xTilde[PP[r, 1]] * alpha + x[PP[r, 1]] * (1 - alpha)
        y_PP[r, 0] = p0 - p1
        r_PP[r, 0] = ti.Matrix.zero(real, dim)

        p0, p1 = x[PP[r, 0]], x[PP[r, 1]]
        Q_PP[r, 0] = min(max(ti.sqrt(PP_hessian(ti.Matrix.zero(real, dim), p0 - p1, dHat2, kappa).norm()), min_Q), max_Q)
    ############################################### PE ###############################################
    for r in range(n_PE[None]):
        p = xTilde[PE[r, 0]] * alpha + x[PE[r, 0]] * (1 - alpha)
        e0 = xTilde[PE[r, 1]] * alpha + x[PE[r, 1]] * (1 - alpha)
        e1 = xTilde[PE[r, 2]] * alpha + x[PE[r, 2]] * (1 - alpha)
        y_PE[r, 0], y_PE[r, 1] = p - e0, p - e1
        r_PE[r, 0], r_PE[r, 1] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)

        p, e0, e1 = x[PE[r, 0]], x[PE[r, 1]], x[PE[r, 2]]
        Q_PE[r, 0] = min(max(ti.sqrt(PE_hessian(ti.Matrix.zero(real, dim), p - e0, p - e1, dHat2, kappa).norm()), min_Q), max_Q)
    ############################################### PT ###############################################
    if ti.static(dim == 3):
        for r in range(n_PT[None]):
            p = xTilde[PT[r, 0]] * alpha + x[PT[r, 0]] * (1 - alpha)
            t0 = xTilde[PT[r, 1]] * alpha + x[PT[r, 1]] * (1 - alpha)
            t1 = xTilde[PT[r, 2]] * alpha + x[PT[r, 2]] * (1 - alpha)
            t2 = xTilde[PT[r, 3]] * alpha + x[PT[r, 3]] * (1 - alpha)
            y_PT[r, 0], y_PT[r, 1], y_PT[r, 2] = p - t0, p - t1, p - t2
            r_PT[r, 0], r_PT[r, 1], r_PT[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)

            p, t0, t1, t2 = x[PT[r, 0]], x[PT[r, 1]], x[PT[r, 2]], x[PT[r, 3]]
            Q_PT[r, 0] = min(max(ti.sqrt(PT_hessian(ti.Matrix.zero(real, dim), p - t0, p - t1, p - t2, dHat2, kappa).norm()), min_Q), max_Q)
    ############################################### EE ###############################################
    if ti.static(dim == 3):
        for r in range(n_EE[None]):
            a0 = xTilde[EE[r, 0]] * alpha + x[EE[r, 0]] * (1 - alpha)
            a1 = xTilde[EE[r, 1]] * alpha + x[EE[r, 1]] * (1 - alpha)
            b0 = xTilde[EE[r, 2]] * alpha + x[EE[r, 2]] * (1 - alpha)
            b1 = xTilde[EE[r, 3]] * alpha + x[EE[r, 3]] * (1 - alpha)
            y_EE[r, 0], y_EE[r, 1], y_EE[r, 2] = a0 - a1, a0 - b0, a0 - b1
            r_EE[r, 0], r_EE[r, 1], r_EE[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)

            a0, a1, b0, b1 = x[EE[r, 0]], x[EE[r, 1]], x[EE[r, 2]], x[EE[r, 3]]
            Q_EE[r, 0] = min(max(ti.sqrt(EE_hessian(ti.Matrix.zero(real, dim), a0 - a1, a0 - b0, a0 - b1, dHat2, kappa).norm()), min_Q), max_Q)
    ############################################### EEM ###############################################
    if ti.static(dim == 3):
        for r in range(n_EEM[None]):
            a0 = xTilde[EEM[r, 0]] * alpha + x[EEM[r, 0]] * (1 - alpha)
            a1 = xTilde[EEM[r, 1]] * alpha + x[EEM[r, 1]] * (1 - alpha)
            b0 = xTilde[EEM[r, 2]] * alpha + x[EEM[r, 2]] * (1 - alpha)
            b1 = xTilde[EEM[r, 3]] * alpha + x[EEM[r, 3]] * (1 - alpha)
            y_EEM[r, 0], y_EEM[r, 1], y_EEM[r, 2] = a0 - a1, a0 - b0, a0 - b1
            r_EEM[r, 0], r_EEM[r, 1], r_EEM[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)

            a0, a1, b0, b1 = x[EEM[r, 0]], x[EEM[r, 1]], x[EEM[r, 2]], x[EEM[r, 3]]
            _a0, _a1, _b0, _b1 = x0[EEM[r, 0]], x0[EEM[r, 1]], x0[EEM[r, 2]], x0[EEM[r, 3]]
            Q_EEM[r, 0] = min(max(ti.sqrt(EEM_hessian(ti.Matrix.zero(real, dim), a0 - a1, a0 - b0, a0 - b1, _a0, _a1, _b0, _b1, dHat2, kappa).norm()), min_Q), max_Q)
    ############################################### PPM ###############################################
    if ti.static(dim == 3):
        for r in range(n_PPM[None]):
            a0 = xTilde[PPM[r, 0]] * alpha + x[PPM[r, 0]] * (1 - alpha)
            a1 = xTilde[PPM[r, 1]] * alpha + x[PPM[r, 1]] * (1 - alpha)
            b0 = xTilde[PPM[r, 2]] * alpha + x[PPM[r, 2]] * (1 - alpha)
            b1 = xTilde[PPM[r, 3]] * alpha + x[PPM[r, 3]] * (1 - alpha)
            y_PPM[r, 0], y_PPM[r, 1], y_PPM[r, 2] = a0 - a1, a0 - b0, a0 - b1
            r_PPM[r, 0], r_PPM[r, 1], r_PPM[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)

            a0, a1, b0, b1 = x[PPM[r, 0]], x[PPM[r, 1]], x[PPM[r, 2]], x[PPM[r, 3]]
            _a0, _a1, _b0, _b1 = x0[PPM[r, 0]], x0[PPM[r, 1]], x0[PPM[r, 2]], x0[PPM[r, 3]]
            Q_PPM[r, 0] = min(max(ti.sqrt(PPM_hessian(ti.Matrix.zero(real, dim), a0 - a1, a0 - b0, a0 - b1, _a0, _a1, _b0, _b1, dHat2, kappa).norm()), min_Q), max_Q)
    ############################################### PEM ###############################################
    if ti.static(dim == 3):
        for r in range(n_PEM[None]):
            a0 = xTilde[PEM[r, 0]] * alpha + x[PEM[r, 0]] * (1 - alpha)
            a1 = xTilde[PEM[r, 1]] * alpha + x[PEM[r, 1]] * (1 - alpha)
            b0 = xTilde[PEM[r, 2]] * alpha + x[PEM[r, 2]] * (1 - alpha)
            b1 = xTilde[PEM[r, 3]] * alpha + x[PEM[r, 3]] * (1 - alpha)
            y_PEM[r, 0], y_PEM[r, 1], y_PEM[r, 2] = a0 - a1, a0 - b0, a0 - b1
            r_PEM[r, 0], r_PEM[r, 1], r_PEM[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)

            a0, a1, b0, b1 = x[PEM[r, 0]], x[PEM[r, 1]], x[PEM[r, 2]], x[PEM[r, 3]]
            _a0, _a1, _b0, _b1 = x0[PEM[r, 0]], x0[PEM[r, 1]], x0[PEM[r, 2]], x0[PEM[r, 3]]
            Q_PEM[r, 0] = min(max(ti.sqrt(PEM_hessian(ti.Matrix.zero(real, dim), a0 - a1, a0 - b0, a0 - b1, _a0, _a1, _b0, _b1, dHat2, kappa).norm()), min_Q), max_Q)
    # reuse y, r
    for c in range(old_n_PP[None]):
        for d in range(n_PP[None]):
            if old_PP[c, 0] == PP[d, 0] and old_PP[c, 1] == PP[d, 1]:
                k = 1.
                if ti.static(update_dbdf):
                    k = old_Q_PP[c, 0] / Q_PP[c, 0]
                else:
                    Q_PP[d, 0] = old_Q_PP[c, 0]
                y_PP[d, 0] = old_y_PP[c, 0]
                r_PP[d, 0] = old_r_PP[c, 0] * k
    for c in range(old_n_PE[None]):
        for d in range(n_PE[None]):
            if old_PE[c, 0] == PE[d, 0] and old_PE[c, 1] == PE[d, 1] and old_PE[c, 2] == PE[d, 2]:
                k = 1.
                if ti.static(update_dbdf):
                    k = old_Q_PE[c, 0] / Q_PE[c, 0]
                else:
                    Q_PE[d, 0], Q_PE[d, 1] = old_Q_PE[c, 0], old_Q_PE[c, 1]
                y_PE[d, 0], y_PE[d, 1] = old_y_PE[c, 0], old_y_PE[c, 1]
                r_PE[d, 0], r_PE[d, 1] = old_r_PE[c, 0] * k, old_r_PE[c, 1] * k
    for c in range(old_n_PT[None]):
        for d in range(n_PT[None]):
            if old_PT[c, 0] == PT[d, 0] and old_PT[c, 1] == PT[d, 1] and old_PT[c, 2] == PT[d, 2] and old_PT[c, 3] == PT[d, 3]:
                k = 1.
                if ti.static(update_dbdf):
                    k = old_Q_PT[c, 0] / Q_PT[c, 0]
                else:
                    Q_PT[d, 0], Q_PT[d, 1], Q_PT[d, 2] = old_Q_PT[c, 0], old_Q_PT[c, 1], old_Q_PT[c, 2]
                y_PT[d, 0], y_PT[d, 1], y_PT[d, 2] = old_y_PT[c, 0], old_y_PT[c, 1], old_y_PT[c, 2]
                r_PT[d, 0], r_PT[d, 1], r_PT[d, 2] = old_r_PT[c, 0] * k, old_r_PT[c, 1] * k, old_r_PT[c, 2] * k
    for c in range(old_n_EE[None]):
        for d in range(n_EE[None]):
            if old_EE[c, 0] == EE[d, 0] and old_EE[c, 1] == EE[d, 1] and old_EE[c, 2] == EE[d, 2] and old_EE[c, 3] == EE[d, 3]:
                k = 1.
                if ti.static(update_dbdf):
                    k = old_Q_EE[c, 0] / Q_EE[c, 0]
                else:
                    Q_EE[d, 0], Q_EE[d, 1], Q_EE[d, 2] = old_Q_EE[c, 0], old_Q_EE[c, 1], old_Q_EE[c, 2]
                y_EE[d, 0], y_EE[d, 1], y_EE[d, 2] = old_y_EE[c, 0], old_y_EE[c, 1], old_y_EE[c, 2]
                r_EE[d, 0], r_EE[d, 1], r_EE[d, 2] = old_r_EE[c, 0] * k, old_r_EE[c, 1] * k, old_r_EE[c, 2] * k
    for c in range(old_n_EEM[None]):
        for d in range(n_EEM[None]):
            if old_EEM[c, 0] == EEM[d, 0] and old_EEM[c, 1] == EEM[d, 1] and old_EEM[c, 2] == EEM[d, 2] and old_EEM[c, 3] == EEM[d, 3]:
                k = 1.
                if ti.static(update_dbdf):
                    k = old_Q_EEM[c, 0] / Q_EEM[c, 0]
                else:
                    Q_EEM[d, 0], Q_EEM[d, 1], Q_EEM[d, 2] = old_Q_EEM[c, 0], old_Q_EEM[c, 1], old_Q_EEM[c, 2]
                y_EEM[d, 0], y_EEM[d, 1], y_EEM[d, 2] = old_y_EEM[c, 0], old_y_EEM[c, 1], old_y_EEM[c, 2]
                r_EEM[d, 0], r_EEM[d, 1], r_EEM[d, 2] = old_r_EEM[c, 0] * k, old_r_EEM[c, 1] * k, old_r_EEM[c, 2] * k
    for c in range(old_n_PPM[None]):
        for d in range(n_PPM[None]):
            if old_PPM[c, 0] == PPM[d, 0] and old_PPM[c, 1] == PPM[d, 1] and old_PPM[c, 2] == PPM[d, 2] and old_PPM[c, 3] == PPM[d, 3]:
                k = 1.
                if ti.static(update_dbdf):
                    k = old_Q_PPM[c, 0] / Q_PPM[c, 0]
                else:
                    Q_PPM[d, 0], Q_PPM[d, 1], Q_PPM[d, 2] = old_Q_PPM[c, 0], old_Q_PPM[c, 1], old_Q_PPM[c, 2]
                y_PPM[d, 0], y_PPM[d, 1], y_PPM[d, 2] = old_y_PPM[c, 0], old_y_PPM[c, 1], old_y_PPM[c, 2]
                r_PPM[d, 0], r_PPM[d, 1], r_PPM[d, 2] = old_r_PPM[c, 0] * k, old_r_PPM[c, 1] * k, old_r_PPM[c, 2] * k
    for c in range(old_n_PEM[None]):
        for d in range(n_PEM[None]):
            if old_PEM[c, 0] == PEM[d, 0] and old_PEM[c, 1] == PEM[d, 1] and old_PEM[c, 2] == PEM[d, 2] and old_PEM[c, 3] == PEM[d, 3]:
                k = 1.
                if ti.static(update_dbdf):
                    k = old_Q_PEM[c, 0] / Q_PEM[c, 0]
                else:
                    Q_PEM[d, 0], Q_PEM[d, 1], Q_PEM[d, 2] = old_Q_PEM[c, 0], old_Q_PEM[c, 1], old_Q_PEM[c, 2]
                y_PEM[d, 0], y_PEM[d, 1], y_PEM[d, 2] = old_y_PEM[c, 0], old_y_PEM[c, 1], old_y_PEM[c, 2]
                r_PEM[d, 0], r_PEM[d, 1], r_PEM[d, 2] = old_r_PEM[c, 0] * k, old_r_PEM[c, 1] * k, old_r_PEM[c, 2] * k


@ti.kernel
def update_dpdf_and_dbdf():
    for i in range(n_elements):
        vol0 = restT[i].determinant() / dim / (dim - 1)
        F = compute_T(i, x) @ restT[i].inverse()
        U, sig, V = svd(F)
        old_W = W[i]
        new_W = ti.sqrt(elasticity_hessian(sig, la, mu).norm() * dt * dt * vol0)
        W[i] = new_W
        u[i] *= (old_W / new_W)


@ti.kernel
def compute_v():
    for i in range(n_particles):
        v[i] = (x[i] - xn[i]) / dt


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
        start = time()
        x.from_numpy(mesh_particles.astype(np.float64)[:, :dim])
        v.fill(0)
        vertices.from_numpy(mesh_elements.astype(np.int32))
        boundary_points.from_numpy(np.array(list(boundary_points_)).astype(np.int32))
        boundary_edges.from_numpy(boundary_edges_.astype(np.int32))
        boundary_triangles.from_numpy(boundary_triangles_.astype(np.int32))
        compute_restT_and_m()
        kappa = compute_adaptive_kappa()
        spatial_hash_inv_dx = 1.0 / compute_mean_of_boundary_edges()
        vertices_ = vertices.to_numpy()
        write_image(0)
        f_start = 0
        if settings['start_frame'] > 0:
            f_start = settings['start_frame']
            [x_, v_] = pickle.load(open(directory + f'caches/{f_start:06d}.p', 'rb'))
            x.from_numpy(x_)
            v.from_numpy(v_)
        for f in range(f_start, 120):
            with Timer("Time Step"):
                print("==================== Frame: ", f, " ====================")
                with Timer("Initialization for xTilde, elasticity"):
                    initial_guess()
                for step in range(20):
                    with Timer("Global Initialization"):
                        alpha = compute_filter(x, xTilde) if step == 0 else 0.0
                        backup_admm_variables()
                        if dim == 2:
                            find_constraints_2D_PE()
                        else:
                            grid.deactivate_all()
                            find_constraints_3D_PT()
                            grid.deactivate_all()
                            find_constraints_3D_EE()
                        remove_duplicated_constraints()
                        reuse_admm_variables(alpha)
                        if update_dpdf:
                            update_dpdf_and_dbdf()

                    with Timer("Global Build System"):
                        data_rhs.fill(0)
                        data_x.fill(0)
                        cnt[None] = 0
                        global_step(_data_row, _data_col, _data_val)
                        global_PP(_data_row, _data_col, _data_val)
                        global_PE(_data_row, _data_col, _data_val)
                        if dim == 3:
                            global_PT(_data_row, _data_col, _data_val)
                            global_EE(_data_row, _data_col, _data_val)
                            global_EEM(_data_row, _data_col, _data_val)
                            global_PPM(_data_row, _data_col, _data_val)
                            global_PEM(_data_row, _data_col, _data_val)
                    with Timer("Global Solve"):
                        solve_system(f * dt)

                    with Timer("Local Step"):
                        local_elasticity()
                        local_PP()
                        local_PE()
                        if dim == 3:
                            local_PT()
                            local_EE()
                            local_EEM()
                            local_PPM()
                            local_PEM()

                    with Timer("Dual Step"):
                        dual_step()
                    print(f, '/', step, ': ', sha1(x.to_numpy()).hexdigest())
                    print('')

                # iters = range(len(prs))
                # fig = plt.figure()
                # plt.plot(iters, prs)
                # plt.title("log primal")
                # fig.savefig(directory + str(f) + "_primal.png")
                # fig = plt.figure()
                # plt.plot(iters, drs)
                # plt.title("log dual")
                # fig.savefig(directory + str(f) + "_dual.png")

                compute_v()
                # TODO: why is visualization so slow?
            with Timer("Visualization"):
                write_image(f + 1)
            pickle.dump([x.to_numpy(), v.to_numpy()], open(directory + f'caches/{f + 1:06d}.p', 'wb'))
            Timer_Print()
        end = time()
        print("!!!!!!!!!!!!!!!!!!!!!!!! ", end - start)
        # cmd = 'ffmpeg -framerate 36 -i "' + directory + 'images/%6d.png" -c:v libx264 -profile:v high -crf 10 -pix_fmt yuv420p -threads 20 ' + directory + 'video.mp4'
        # os.system((cmd))
