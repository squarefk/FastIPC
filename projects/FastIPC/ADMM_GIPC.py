from reader import *
from common.math.gipc import *
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
rc = ti.field(real, shape=n_particles * dim)

MAX_C = 100000
GPE = ti.field(ti.i32, shape=(MAX_C, 3))
n_GPE = ti.field(ti.i32, shape=())
y_GPE, r_GPE, Q_GPE = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 2)).place(y_GPE, r_GPE, Q_GPE)
GPT = ti.field(ti.i32, shape=(MAX_C, 4))
n_GPT = ti.field(ti.i32, shape=())
y_GPT, r_GPT, Q_GPT = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_GPT, r_GPT, Q_GPT)
GEE = ti.field(ti.i32, shape=(MAX_C, 4))
n_GEE = ti.field(ti.i32, shape=())
y_GEE, r_GEE, Q_GEE = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_GEE, r_GEE, Q_GEE)
GEEM = ti.field(ti.i32, shape=(MAX_C, 4))
n_GEEM = ti.field(ti.i32, shape=())
y_GEEM, r_GEEM, Q_GEEM = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_GEEM, r_GEEM, Q_GEEM)

old_GPE = ti.field(ti.i32, shape=(MAX_C, 3))
old_n_GPE = ti.field(ti.i32, shape=())
old_y_GPE, old_r_GPE, old_Q_GPE = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 2)).place(old_y_GPE, old_r_GPE, old_Q_GPE)
old_GPT = ti.field(ti.i32, shape=(MAX_C, 4))
old_n_GPT = ti.field(ti.i32, shape=())
old_y_GPT, old_r_GPT, old_Q_GPT = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_GPT, old_r_GPT, old_Q_GPT)
old_GEE = ti.field(ti.i32, shape=(MAX_C, 4))
old_n_GEE = ti.field(ti.i32, shape=())
old_y_GEE, old_r_GEE, old_Q_GEE = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_GEE, old_r_GEE, old_Q_GEE)
old_GEEM = ti.field(ti.i32, shape=(MAX_C, 4))
old_n_GEEM = ti.field(ti.i32, shape=())
old_y_GEEM, old_r_GEEM, old_Q_GEEM = vec(), vec(), scalar()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_GEEM, old_r_GEEM, old_Q_GEEM)

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
        W[i] = ti.sqrt(elasticity_hessian(sig, la, mu).norm() * dt * dt * vol0)
        # W[i] = ti.sqrt((la + mu * 2 / 3) * dt * dt * vol0)
        z[i] = currentT @ restT[i].inverse()
        u[i] = ti.Matrix.zero(real, dim, dim)
    n_GPE[None], n_GPT[None], n_GEE[None], n_GEEM[None] = 0, 0, 0, 0


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
def global_GPE(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
    ETE3 = ti.Matrix([[2, -1, -1], [-1, 1, 0], [-1, 0, 1]])
    for _ in range(1):
        for c in range(n_GPE[None]):
            Q = Q_GPE[c, 0]
            for p in ti.static(range(3)):
                for q in ti.static(range(3)):
                    idx = cnt[None] + c * 9 + p * 3 + q
                    data_row[idx] = GPE[c, p]
                    data_col[idx] = GPE[c, q]
                    data_val[idx] = ETE3[p, q] * Q * Q
            for j in ti.static(range(dim)):
                data_rhs[GPE[c, 0] * dim + j] += (y_GPE(j)[c, 0] - r_GPE(j)[c, 0]) * Q * Q
                data_rhs[GPE[c, 0] * dim + j] += (y_GPE(j)[c, 1] - r_GPE(j)[c, 1]) * Q * Q
                data_rhs[GPE[c, 1] * dim + j] -= (y_GPE(j)[c, 0] - r_GPE(j)[c, 0]) * Q * Q
                data_rhs[GPE[c, 2] * dim + j] -= (y_GPE(j)[c, 1] - r_GPE(j)[c, 1]) * Q * Q
    cnt[None] += n_GPE[None] * 9
@ti.kernel
def global_GPT(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for _ in range(1):
        for c in range(n_GPT[None]):
            Q = Q_GPT[c, 0]
            for p in ti.static(range(4)):
                for q in ti.static(range(4)):
                    idx = cnt[None] + c * 16 + p * 4 + q
                    data_row[idx] = GPT[c, p]
                    data_col[idx] = GPT[c, q]
                    data_val[idx] = ETE4[p, q] * Q * Q
            for j in ti.static(range(3)):
                data_rhs[GPT[c, 0] * 3 + j] += (y_GPT(j)[c, 0] - r_GPT(j)[c, 0]) * Q * Q
                data_rhs[GPT[c, 0] * 3 + j] += (y_GPT(j)[c, 1] - r_GPT(j)[c, 1]) * Q * Q
                data_rhs[GPT[c, 0] * 3 + j] += (y_GPT(j)[c, 2] - r_GPT(j)[c, 2]) * Q * Q
                data_rhs[GPT[c, 1] * 3 + j] -= (y_GPT(j)[c, 0] - r_GPT(j)[c, 0]) * Q * Q
                data_rhs[GPT[c, 2] * 3 + j] -= (y_GPT(j)[c, 1] - r_GPT(j)[c, 1]) * Q * Q
                data_rhs[GPT[c, 3] * 3 + j] -= (y_GPT(j)[c, 2] - r_GPT(j)[c, 2]) * Q * Q
    cnt[None] += n_GPT[None] * 16
@ti.kernel
def global_GEE(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for _ in range(1):
        for c in range(n_GEE[None]):
            Q = Q_GEE[c, 0]
            for p in ti.static(range(4)):
                for q in ti.static(range(4)):
                    idx = cnt[None] + c * 16 + p * 4 + q
                    data_row[idx] = GEE[c, p]
                    data_col[idx] = GEE[c, q]
                    data_val[idx] = ETE4[p, q] * Q * Q
            for j in ti.static(range(3)):
                data_rhs[GEE[c, 0] * 3 + j] += (y_GEE(j)[c, 0] - r_GEE(j)[c, 0]) * Q * Q
                data_rhs[GEE[c, 0] * 3 + j] += (y_GEE(j)[c, 1] - r_GEE(j)[c, 1]) * Q * Q
                data_rhs[GEE[c, 0] * 3 + j] += (y_GEE(j)[c, 2] - r_GEE(j)[c, 2]) * Q * Q
                data_rhs[GEE[c, 1] * 3 + j] -= (y_GEE(j)[c, 0] - r_GEE(j)[c, 0]) * Q * Q
                data_rhs[GEE[c, 2] * 3 + j] -= (y_GEE(j)[c, 1] - r_GEE(j)[c, 1]) * Q * Q
                data_rhs[GEE[c, 3] * 3 + j] -= (y_GEE(j)[c, 2] - r_GEE(j)[c, 2]) * Q * Q
    cnt[None] += n_GEE[None] * 16
@ti.kernel
def global_GEEM(data_row: ti.ext_arr(), data_col: ti.ext_arr(), data_val: ti.ext_arr()):
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for _ in range(1):
        for c in range(n_GEEM[None]):
            Q = Q_GEEM[c, 0]
            for p in ti.static(range(4)):
                for q in ti.static(range(4)):
                    idx = cnt[None] + c * 16 + p * 4 + q
                    data_row[idx] = GEEM[c, p]
                    data_col[idx] = GEEM[c, q]
                    data_val[idx] = ETE4[p, q] * Q * Q
            for j in ti.static(range(3)):
                data_rhs[GEEM[c, 0] * 3 + j] += (y_GEEM(j)[c, 0] - r_GEEM(j)[c, 0]) * Q * Q
                data_rhs[GEEM[c, 0] * 3 + j] += (y_GEEM(j)[c, 1] - r_GEEM(j)[c, 1]) * Q * Q
                data_rhs[GEEM[c, 0] * 3 + j] += (y_GEEM(j)[c, 2] - r_GEEM(j)[c, 2]) * Q * Q
                data_rhs[GEEM[c, 1] * 3 + j] -= (y_GEEM(j)[c, 0] - r_GEEM(j)[c, 0]) * Q * Q
                data_rhs[GEEM[c, 2] * 3 + j] -= (y_GEEM(j)[c, 1] - r_GEEM(j)[c, 1]) * Q * Q
                data_rhs[GEEM[c, 3] * 3 + j] -= (y_GEEM(j)[c, 2] - r_GEEM(j)[c, 2]) * Q * Q
    cnt[None] += n_GEEM[None] * 16


def solve_system(current_time):
    with Timer("Filter Global Prepare"):
        @ti.kernel
        def before_solve():
            for i in range(n_particles):
                xx[i] = x[i]
            for i in range(n_elements):
                zz[i] = z[i]
        before_solve()

    with Timer("Init DBC"):
        dirichlet_fixed, dirichlet_value = settings['dirichlet'](current_time)
        D, V = dirichlet_fixed, dirichlet_value.reshape((n_particles * dim))
        dfx.from_numpy(D.astype(np.int32))
        dfv.from_numpy(V.astype(np.float64))

    if cnt[None] >= MAX_LINEAR or n_GPE[None] >= MAX_C or n_GPT[None] >= MAX_C or n_GEE[None] >= MAX_C or n_GEEM[None] >= MAX_C:
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
def local_GPE():
    for c in range(n_GPE[None]):
        pos = ti.Matrix.zero(real, dim * 2)
        posTilde = ti.Matrix.zero(real, dim * 2)
        for i in ti.static(range(dim)):
            pos[i] = x(i)[GPE[c, 0]] - x(i)[GPE[c, 1]]
            pos[i + dim] = x(i)[GPE[c, 0]] - x(i)[GPE[c, 2]]
            posTilde[i] = x(i)[GPE[c, 0]] - x(i)[GPE[c, 1]] + r_GPE(i)[c, 0]
            posTilde[i + dim] = x(i)[GPE[c, 0]] - x(i)[GPE[c, 2]] + r_GPE(i)[c, 1]
        Q = Q_GPE[c, 0]
        op = ti.Matrix.zero(real, dim)
        converge = False
        iter = 0
        while not converge:
            g = extract_vec(GPE_gradient(op, extract_vec(pos, list(range(0, dim))), extract_vec(pos, list(range(dim, dim * 2))), dHat2, kappa), [2, 3, 4, 5]) + (pos - posTilde) * Q * Q
            P = project_pd(extract_mat(GPE_hessian(op, extract_vec(pos, list(range(0, dim))), extract_vec(pos, list(range(dim, dim * 2))), dHat2, kappa), [2, 3, 4, 5])) + ti.Matrix.identity(real, dim * 2) * Q * Q
            p = -solve(P, g)
            iter += 1
            if p.norm_sqr() < 1e-6:
                converge = True
            if iter & 31 == 0:
                print("local PE iters:", iter, ", residual:", p.norm_sqr())
            alpha = point_edge_ccd(ti.Vector([0.0, 0.0]), ti.Vector([pos[0], pos[1]]), ti.Vector([pos[2], pos[3]]), ti.Vector([0.0, 0.0]), ti.Vector([p[0], p[1]]), ti.Vector([p[2], p[3]]), 0.1)
            pos0 = pos
            E0 = GPE_energy(op, extract_vec(pos, list(range(0, dim))), extract_vec(pos, list(range(dim, dim * 2))), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            pos = pos0 + alpha * p
            E = GPE_energy(op, extract_vec(pos, list(range(0, dim))), extract_vec(pos, list(range(dim, dim * 2))), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = GPE_energy(op, extract_vec(pos, list(range(0, dim))), extract_vec(pos, list(range(dim, dim * 2))), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
        for i in ti.static(range(dim)):
            y_GPE(i)[c, 0] = pos[i]
            y_GPE(i)[c, 1] = pos[i + dim]
@ti.kernel
def local_GPT():
    for c in range(n_GPT[None]):
        pos = ti.Vector([x(0)[GPT[c, 0]] - x(0)[GPT[c, 1]], x(1)[GPT[c, 0]] - x(1)[GPT[c, 1]], x(2)[GPT[c, 0]] - x(2)[GPT[c, 1]],
                         x(0)[GPT[c, 0]] - x(0)[GPT[c, 2]], x(1)[GPT[c, 0]] - x(1)[GPT[c, 2]], x(2)[GPT[c, 0]] - x(2)[GPT[c, 2]],
                         x(0)[GPT[c, 0]] - x(0)[GPT[c, 3]], x(1)[GPT[c, 0]] - x(1)[GPT[c, 3]], x(2)[GPT[c, 0]] - x(2)[GPT[c, 3]]])
        posTilde = ti.Vector([x(0)[GPT[c, 0]] - x(0)[GPT[c, 1]] + r_GPT(0)[c, 0], x(1)[GPT[c, 0]] - x(1)[GPT[c, 1]] + r_GPT(1)[c, 0], x(2)[GPT[c, 0]] - x(2)[GPT[c, 1]] + r_GPT(2)[c, 0],
                              x(0)[GPT[c, 0]] - x(0)[GPT[c, 2]] + r_GPT(0)[c, 1], x(1)[GPT[c, 0]] - x(1)[GPT[c, 2]] + r_GPT(1)[c, 1], x(2)[GPT[c, 0]] - x(2)[GPT[c, 2]] + r_GPT(2)[c, 1],
                              x(0)[GPT[c, 0]] - x(0)[GPT[c, 3]] + r_GPT(0)[c, 2], x(1)[GPT[c, 0]] - x(1)[GPT[c, 3]] + r_GPT(1)[c, 2], x(2)[GPT[c, 0]] - x(2)[GPT[c, 3]] + r_GPT(2)[c, 2]])
        Q = Q_GPT[c, 0]
        op = ti.Matrix.zero(real, dim)
        converge = False
        iter = 0
        while not converge:
            g = extract_vec(GPT_gradient(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa), [3, 4, 5, 6, 7, 8, 9, 10, 11]) + (pos - posTilde) * Q * Q
            P = project_pd(extract_mat(GPT_hessian(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa), [3, 4, 5, 6, 7, 8, 9, 10, 11])) + ti.Matrix.identity(real, dim * 3) * Q * Q
            p = -solve(P, g)
            iter += 1
            if p.norm_sqr() < 1e-6:
                converge = True
            if iter & 31 == 0:
                print("local GPT iters:", iter, ", residual:", p.norm_sqr())

            _p, _t0, _t1, _t2 = ti.Vector([0., 0., 0.]), ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
            _dp, _dt0, _dt1, _dt2 = ti.Vector([0., 0., 0.]), ti.Vector([p[0], p[1], p[2]]), ti.Vector([p[3], p[4], p[5]]), ti.Vector([p[6], p[7], p[8]])
            dist2 = PT_dist2(_p, _t0, _t1, _t2, PT_type(_p, _t0, _t1, _t2))
            alpha = point_triangle_ccd(_p, _t0, _t1, _t2, _dp, _dt0, _dt1, _dt2, 0.2, dist2)

            pos0 = pos
            E0 = GPT_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            pos = pos0 + alpha * p
            E = GPT_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = GPT_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
        y_GPT[c, 0], y_GPT[c, 1], y_GPT[c, 2] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
@ti.kernel
def local_GEE():
    for c in range(n_GEE[None]):
        pos = ti.Vector([x(0)[GEE[c, 0]] - x(0)[GEE[c, 1]], x(1)[GEE[c, 0]] - x(1)[GEE[c, 1]], x(2)[GEE[c, 0]] - x(2)[GEE[c, 1]],
                         x(0)[GEE[c, 0]] - x(0)[GEE[c, 2]], x(1)[GEE[c, 0]] - x(1)[GEE[c, 2]], x(2)[GEE[c, 0]] - x(2)[GEE[c, 2]],
                         x(0)[GEE[c, 0]] - x(0)[GEE[c, 3]], x(1)[GEE[c, 0]] - x(1)[GEE[c, 3]], x(2)[GEE[c, 0]] - x(2)[GEE[c, 3]]])
        posTilde = ti.Vector([x(0)[GEE[c, 0]] - x(0)[GEE[c, 1]] + r_GEE(0)[c, 0], x(1)[GEE[c, 0]] - x(1)[GEE[c, 1]] + r_GEE(1)[c, 0], x(2)[GEE[c, 0]] - x(2)[GEE[c, 1]] + r_GEE(2)[c, 0],
                              x(0)[GEE[c, 0]] - x(0)[GEE[c, 2]] + r_GEE(0)[c, 1], x(1)[GEE[c, 0]] - x(1)[GEE[c, 2]] + r_GEE(1)[c, 1], x(2)[GEE[c, 0]] - x(2)[GEE[c, 2]] + r_GEE(2)[c, 1],
                              x(0)[GEE[c, 0]] - x(0)[GEE[c, 3]] + r_GEE(0)[c, 2], x(1)[GEE[c, 0]] - x(1)[GEE[c, 3]] + r_GEE(1)[c, 2], x(2)[GEE[c, 0]] - x(2)[GEE[c, 3]] + r_GEE(2)[c, 2]])
        Q = Q_GEE[c, 0]
        op = ti.Matrix.zero(real, dim)
        converge = False
        iter = 0
        while not converge:
            g = extract_vec(GEE_gradient(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa), [3, 4, 5, 6, 7, 8, 9, 10, 11]) + (pos - posTilde) * Q * Q
            P = project_pd(extract_mat(GEE_hessian(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa), [3, 4, 5, 6, 7, 8, 9, 10, 11])) + ti.Matrix.identity(real, dim * 3) * Q * Q
            p = -solve(P, g)
            iter += 1
            if p.norm_sqr() < 1e-6:
                converge = True
            if iter & 31 == 0:
                print("local GEE iters:", iter, ", residual:", p.norm_sqr())

            _a0, _a1, _b0, _b1 = ti.Vector([0., 0., 0.]), ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
            _da0, _da1, _db0, _db1 = ti.Vector([0., 0., 0.]), ti.Vector([p[0], p[1], p[2]]), ti.Vector([p[3], p[4], p[5]]), ti.Vector([p[6], p[7], p[8]])
            dist2 = PT_dist2(_a0, _a1, _b0, _b1, PT_type(_a0, _a1, _b0, _b1))
            alpha = edge_edge_ccd(_a0, _a1, _b0, _b1, _da0, _da1, _db0, _db1, 0.2, dist2)

            pos0 = pos
            E0 = GEE_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            pos = pos0 + alpha * p
            E = GEE_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = GEE_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
        y_GEE[c, 0], y_GEE[c, 1], y_GEE[c, 2] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
@ti.kernel
def local_GEEM():
    for c in range(n_GEEM[None]):
        pos = ti.Vector([x(0)[GEEM[c, 0]] - x(0)[GEEM[c, 1]], x(1)[GEEM[c, 0]] - x(1)[GEEM[c, 1]], x(2)[GEEM[c, 0]] - x(2)[GEEM[c, 1]],
                         x(0)[GEEM[c, 0]] - x(0)[GEEM[c, 2]], x(1)[GEEM[c, 0]] - x(1)[GEEM[c, 2]], x(2)[GEEM[c, 0]] - x(2)[GEEM[c, 2]],
                         x(0)[GEEM[c, 0]] - x(0)[GEEM[c, 3]], x(1)[GEEM[c, 0]] - x(1)[GEEM[c, 3]], x(2)[GEEM[c, 0]] - x(2)[GEEM[c, 3]]])
        posTilde = ti.Vector([x(0)[GEEM[c, 0]] - x(0)[GEEM[c, 1]] + r_GEEM(0)[c, 0], x(1)[GEEM[c, 0]] - x(1)[GEEM[c, 1]] + r_GEEM(1)[c, 0], x(2)[GEEM[c, 0]] - x(2)[GEEM[c, 1]] + r_GEEM(2)[c, 0],
                              x(0)[GEEM[c, 0]] - x(0)[GEEM[c, 2]] + r_GEEM(0)[c, 1], x(1)[GEEM[c, 0]] - x(1)[GEEM[c, 2]] + r_GEEM(1)[c, 1], x(2)[GEEM[c, 0]] - x(2)[GEEM[c, 2]] + r_GEEM(2)[c, 1],
                              x(0)[GEEM[c, 0]] - x(0)[GEEM[c, 3]] + r_GEEM(0)[c, 2], x(1)[GEEM[c, 0]] - x(1)[GEEM[c, 3]] + r_GEEM(1)[c, 2], x(2)[GEEM[c, 0]] - x(2)[GEEM[c, 3]] + r_GEEM(2)[c, 2]])
        Q = Q_GEEM[c, 0]
        op = ti.Matrix.zero(real, dim)
        _a0, _a1, _b0, _b1 = x0[GEEM[c, 0]], x0[GEEM[c, 1]], x0[GEEM[c, 2]], x0[GEEM[c, 3]]
        converge = False
        iter = 0
        while not converge:
            g = extract_vec(GEEM_gradient(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa), [3, 4, 5, 6, 7, 8, 9, 10, 11]) + (pos - posTilde) * Q * Q
            P = project_pd(extract_mat(GEEM_hessian(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa), [3, 4, 5, 6, 7, 8, 9, 10, 11])) + ti.Matrix.identity(real, dim * 3) * Q * Q
            p = -solve(P, g)
            iter += 1
            if p.norm_sqr() < 1e-6:
                converge = True
            if iter & 31 == 0:
                print("local GEEM iters:", iter, ", residual:", p.norm_sqr())

            _a0, _a1, _b0, _b1 = ti.Vector([0., 0., 0.]), ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
            _da0, _da1, _db0, _db1 = ti.Vector([0., 0., 0.]), ti.Vector([p[0], p[1], p[2]]), ti.Vector([p[3], p[4], p[5]]), ti.Vector([p[6], p[7], p[8]])
            dist2 = PT_dist2(_a0, _a1, _b0, _b1, PT_type(_a0, _a1, _b0, _b1))
            alpha = edge_edge_ccd(_a0, _a1, _b0, _b1, _da0, _da1, _db0, _db1, 0.2, dist2)

            pos0 = pos
            E0 = GEEM_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            pos = pos0 + alpha * p
            E = GEEM_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = GEEM_energy(op, ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]]), _a0, _a1, _b0, _b1, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
        y_GEEM[c, 0], y_GEEM[c, 1], y_GEEM[c, 2] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])


@ti.kernel
def prime_residual() -> real:
    residual = 0.0
    for i in range(n_elements):
        currentT = compute_T(i, x)
        F = currentT @ restT[i].inverse()
        residual += (F - z[i]).norm_sqr() * W[i] * W[i]
    for c in range(n_GPE[None]):
        Q = Q_GPE[c, 0]
        residual += (x[GPE[c, 0]] - x[GPE[c, 1]] - y_GPE[c, 0]).norm_sqr() * Q * Q
        residual += (x[GPE[c, 0]] - x[GPE[c, 2]] - y_GPE[c, 1]).norm_sqr() * Q * Q
    return residual


@ti.kernel
def dual_residual() -> real:
    residual = 0.0
    for i in rc:
        rc[i] = 0
    for e in range(n_elements):
        A = restT[e].inverse()
        delta = z[e] - zz[e]
        for p in ti.static(range(dim + 1)):
            for i in ti.static(range(dim)):
                for j in ti.static(range(dim)):
                    q = i
                    rc[vertices[e, p] * 2 + q] += X2F(p, q, i, j, A) * delta[i, j] * W[e] * W[e]
    for i in rc:
        residual += rc[i] * rc[i]
        rc[i] = 0
    for c in range(old_n_GPE[None]):
        Q = old_Q_GPE[c, 0]
        exist = False
        for d in range(n_GPE[None]):
            if old_GPE[c, 0] == GPE[d, 0] and old_GPE[c, 1] == GPE[d, 1] and old_GPE[c, 2] == GPE[d, 2]:
                exist = True
                for j in ti.static(range(dim)):
                    rc[GPE[d, 0] * dim + j] += (y_GPE(j)[d, 0] - old_y_GPE(j)[c, 0]) * Q * Q
                    rc[GPE[d, 0] * dim + j] += (y_GPE(j)[d, 1] - old_y_GPE(j)[c, 1]) * Q * Q
                    rc[GPE[d, 1] * dim + j] -= (y_GPE(j)[d, 0] - old_y_GPE(j)[c, 0]) * Q * Q
                    rc[GPE[d, 2] * dim + j] -= (y_GPE(j)[d, 1] - old_y_GPE(j)[c, 1]) * Q * Q
        if not exist:
            y0 = x[old_GPE[c, 0]] - x[old_GPE[c, 1]]
            y1 = x[old_GPE[c, 0]] - x[old_GPE[c, 2]]
            for j in ti.static(range(dim)):
                rc[old_GPE[c, 0] * dim + j] += (y0(j) - old_y_GPE(j)[c, 0]) * Q * Q
                rc[old_GPE[c, 0] * dim + j] += (y1(j) - old_y_GPE(j)[c, 1]) * Q * Q
                rc[old_GPE[c, 1] * dim + j] -= (y0(j) - old_y_GPE(j)[c, 0]) * Q * Q
                rc[old_GPE[c, 2] * dim + j] -= (y1(j) - old_y_GPE(j)[c, 1]) * Q * Q
    for d in range(n_GPE[None]):
        Q = Q_GPE[d, 0]
        exist = False
        for c in range(old_n_GPE[None]):
            if old_GPE[c, 0] == GPE[d, 0] and old_GPE[c, 1] == GPE[d, 1] and old_GPE[c, 2] == GPE[d, 2]:
                exist = True
        if not exist:
            y0 = xx[GPE[d, 0]] - xx[GPE[d, 1]]
            y1 = xx[GPE[d, 0]] - xx[GPE[d, 2]]
            for j in ti.static(range(dim)):
                rc[GPE[d, 0] * dim + j] += (y_GPE(j)[d, 0] - y0(j)) * Q * Q
                rc[GPE[d, 0] * dim + j] += (y_GPE(j)[d, 1] - y1(j)) * Q * Q
                rc[GPE[d, 1] * dim + j] -= (y_GPE(j)[d, 0] - y0(j)) * Q * Q
                rc[GPE[d, 2] * dim + j] -= (y_GPE(j)[d, 1] - y1(j)) * Q * Q
    for i in rc:
        residual += rc[i] * rc[i]
    return residual


@ti.kernel
def newton_gradient_residual() -> real:
    residual = 0.0
    for i in rc:
        rc[i] = 0
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            rc[i * dim + d] += m[i] * (x(d)[i] - xTilde(d)[i])
    for e in range(n_elements):
        A = restT[e].inverse()
        F = compute_T(e, x) @ A
        vol0 = restT[e].determinant() / dim / (dim - 1)
        P = elasticity_first_piola_kirchoff_stress(F, la, mu) * dt * dt * vol0
        for p in ti.static(range(dim + 1)):
            for i in ti.static(range(dim)):
                for j in ti.static(range(dim)):
                    q = i
                    rc[vertices[e, p] * dim + q] += X2F(p, q, i, j, A) * P[i, j]
    for c in range(n_GPE[None]):
        p, e0, e1 = x[GPE[c, 0]], x[GPE[c, 1]], x[GPE[c, 2]]
        g = extract_vec(GPE_gradient(ti.Matrix.zero(real, dim), p - e0, p - e1, dHat2, kappa), [2, 3, 4, 5])
        for j in ti.static(range(dim)):
            rc[GPE[c, 0] * dim + j] += g[0 * dim + j]
            rc[GPE[c, 0] * dim + j] += g[1 * dim + j]
            rc[GPE[c, 1] * dim + j] -= g[0 * dim + j]
            rc[GPE[c, 2] * dim + j] -= g[1 * dim + j]
    for i in rc:
        residual += rc[i] * rc[i]
    return residual

@ti.kernel
def dual_step():
    for i in range(n_elements):
        currentT = compute_T(i, x)
        F = currentT @ restT[i].inverse()
        u[i] += F - z[i]
    for c in range(n_GPE[None]):
        r_GPE[c, 0] += x[GPE[c, 0]] - x[GPE[c, 1]] - y_GPE[c, 0]
        r_GPE[c, 1] += x[GPE[c, 0]] - x[GPE[c, 2]] - y_GPE[c, 1]
    for c in range(n_GPT[None]):
        r_GPT[c, 0] += x[GPT[c, 0]] - x[GPT[c, 1]] - y_GPT[c, 0]
        r_GPT[c, 1] += x[GPT[c, 0]] - x[GPT[c, 2]] - y_GPT[c, 1]
        r_GPT[c, 2] += x[GPT[c, 0]] - x[GPT[c, 3]] - y_GPT[c, 2]
    for c in range(n_GEE[None]):
        r_GEE[c, 0] += x[GEE[c, 0]] - x[GEE[c, 1]] - y_GEE[c, 0]
        r_GEE[c, 1] += x[GEE[c, 0]] - x[GEE[c, 2]] - y_GEE[c, 1]
        r_GEE[c, 2] += x[GEE[c, 0]] - x[GEE[c, 3]] - y_GEE[c, 2]
    for c in range(n_GEEM[None]):
        r_GEEM[c, 0] += x[GEEM[c, 0]] - x[GEEM[c, 1]] - y_GEEM[c, 0]
        r_GEEM[c, 1] += x[GEEM[c, 0]] - x[GEEM[c, 2]] - y_GEEM[c, 1]
        r_GEEM[c, 2] += x[GEEM[c, 0]] - x[GEEM[c, 3]] - y_GEEM[c, 2]


@ti.kernel
def backup_admm_variables():
    old_n_GPE[None] = n_GPE[None]
    for c in range(old_n_GPE[None]):
        old_GPE[c, 0], old_GPE[c, 1], old_GPE[c, 2] = GPE[c, 0], GPE[c, 1], GPE[c, 2]
        old_y_GPE[c, 0], old_y_GPE[c, 1] = y_GPE[c, 0], y_GPE[c, 1]
        old_r_GPE[c, 0], old_r_GPE[c, 1] = r_GPE[c, 0], r_GPE[c, 1]
        old_Q_GPE[c, 0], old_Q_GPE[c, 1] = Q_GPE[c, 0], Q_GPE[c, 1]
    old_n_GPT[None] = n_GPT[None]
    for c in range(old_n_GPT[None]):
        old_GPT[c, 0], old_GPT[c, 1], old_GPT[c, 2], old_GPT[c, 3] = GPT[c, 0], GPT[c, 1], GPT[c, 2], GPT[c, 3]
        old_y_GPT[c, 0], old_y_GPT[c, 1], old_y_GPT[c, 2] = y_GPT[c, 0], y_GPT[c, 1], y_GPT[c, 2]
        old_r_GPT[c, 0], old_r_GPT[c, 1], old_r_GPT[c, 2] = r_GPT[c, 0], r_GPT[c, 1], r_GPT[c, 2]
        old_Q_GPT[c, 0], old_Q_GPT[c, 1], old_Q_GPT[c, 2] = Q_GPT[c, 0], Q_GPT[c, 1], Q_GPT[c, 2]
    old_n_GEE[None] = n_GEE[None]
    for c in range(old_n_GEE[None]):
        old_GEE[c, 0], old_GEE[c, 1], old_GEE[c, 2], old_GEE[c, 3] = GEE[c, 0], GEE[c, 1], GEE[c, 2], GEE[c, 3]
        old_y_GEE[c, 0], old_y_GEE[c, 1], old_y_GEE[c, 2] = y_GEE[c, 0], y_GEE[c, 1], y_GEE[c, 2]
        old_r_GEE[c, 0], old_r_GEE[c, 1], old_r_GEE[c, 2] = r_GEE[c, 0], r_GEE[c, 1], r_GEE[c, 2]
        old_Q_GEE[c, 0], old_Q_GEE[c, 1], old_Q_GEE[c, 2] = Q_GEE[c, 0], Q_GEE[c, 1], Q_GEE[c, 2]
    old_n_GEEM[None] = n_GEEM[None]
    for c in range(old_n_GEEM[None]):
        old_GEEM[c, 0], old_GEEM[c, 1], old_GEEM[c, 2], old_GEEM[c, 3] = GEEM[c, 0], GEEM[c, 1], GEEM[c, 2], GEEM[c, 3]
        old_y_GEEM[c, 0], old_y_GEEM[c, 1], old_y_GEEM[c, 2] = y_GEEM[c, 0], y_GEEM[c, 1], y_GEEM[c, 2]
        old_r_GEEM[c, 0], old_r_GEEM[c, 1], old_r_GEEM[c, 2] = r_GEEM[c, 0], r_GEEM[c, 1], r_GEEM[c, 2]
        old_Q_GEEM[c, 0], old_Q_GEEM[c, 1], old_Q_GEEM[c, 2] = Q_GEEM[c, 0], Q_GEEM[c, 1], Q_GEEM[c, 2]


@ti.kernel
def find_constraints_2D_PE():
    for i in range(n_boundary_points):
        p = boundary_points[i]
        for j in range(n_boundary_edges):
            e0 = boundary_edges[j, 0]
            e1 = boundary_edges[j, 1]
            if p != e0 and p != e1 and point_edge_ccd_broadphase(x[p], x[e0], x[e1], dHat):
                case = PE_type(x[p], x[e0], x[e1])
                if PE_dist2(x[p], x[e0], x[e1], case) < dHat2:
                    n = ti.atomic_add(n_GPE[None], 1)
                    GPE[n, 0], GPE[n, 1], GPE[n, 2] = p, e0, e1

@ti.func
def attempt_PT(p, t0, t1, t2):
    if p != t0 and p != t1 and p != t2 and point_triangle_ccd_broadphase(x[p], x[t0], x[t1], x[t2], dHat):
        case = PT_type(x[p], x[t0], x[t1], x[t2])
        if PT_dist2(x[p], x[t0], x[t1], x[t2], case) < dHat2:
            n = ti.atomic_add(n_GPT[None], 1)
            GPT[n, 0], GPT[n, 1], GPT[n, 2], GPT[n, 3] = p, t0, t1, t2
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
        if EE_dist2(x[a0], x[a1], x[b0], x[b1], case) < dHat2:
            if EECN2 < eps_x:
                n = ti.atomic_add(n_GEEM[None], 1)
                GEEM[n, 0], GEEM[n, 1], GEEM[n, 2], GEEM[n, 3] = a0, a1, b0, b1
            else:
                n = ti.atomic_add(n_GEE[None], 1)
                GEE[n, 0], GEE[n, 1], GEE[n, 2], GEE[n, 3] = a0, a1, b0, b1
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
    tmp = np.unique(GPE.to_numpy()[:n_GPE[None], :], axis=0)
    n_GPE[None] = len(tmp)
    GPE.from_numpy(np.resize(tmp, (MAX_C, 3)))
    tmp = np.unique(GPT.to_numpy()[:n_GPT[None], :], axis=0)
    n_GPT[None] = len(tmp)
    GPT.from_numpy(np.resize(tmp, (MAX_C, 4)))
    tmp = np.unique(GEE.to_numpy()[:n_GEE[None], :], axis=0)
    n_GEE[None] = len(tmp)
    GEE.from_numpy(np.resize(tmp, (MAX_C, 4)))
    tmp = np.unique(GEEM.to_numpy()[:n_GEEM[None], :], axis=0)
    n_GEEM[None] = len(tmp)
    GEEM.from_numpy(np.resize(tmp, (MAX_C, 4)))
    print("Find constraints: ", n_GPE[None], n_GPT[None], n_GEE[None], n_GEEM[None])


@ti.kernel
def reuse_admm_variables(alpha: real):
    # xTilde initiated y, r
    min_Q = ti.sqrt(PP_hessian(ti.Matrix.zero(real, dim), ti.Matrix.one(real, dim) * 9e-1 * dHat, dHat2, kappa).norm()) / 10
    max_Q = ti.sqrt(PP_hessian(ti.Matrix.zero(real, dim), ti.Matrix.one(real, dim) * 1e-4 * dHat, dHat2, kappa).norm()) * 10
    ############################################### PE ###############################################
    if ti.static(dim == 2):
        for r in range(n_GPE[None]):
            p = xTilde[GPE[r, 0]] * alpha + x[GPE[r, 0]] * (1 - alpha)
            e0 = xTilde[GPE[r, 1]] * alpha + x[GPE[r, 1]] * (1 - alpha)
            e1 = xTilde[GPE[r, 2]] * alpha + x[GPE[r, 2]] * (1 - alpha)
            y_GPE[r, 0], y_GPE[r, 1] = p - e0, p - e1
            r_GPE[r, 0], r_GPE[r, 1] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)

            p, e0, e1 = x[GPE[r, 0]], x[GPE[r, 1]], x[GPE[r, 2]]
            # Q_GPE[r, 0] = min(max(ti.sqrt(GPE_hessian(ti.Matrix.zero(real, dim), p - e0, p - e1, dHat2, kappa).norm()), min_Q), max_Q)
            Q_GPE[r, 0] = ti.sqrt(GPE_hessian(ti.Matrix.zero(real, dim), p - e0, p - e1, dHat2, kappa).norm())
    ############################################### PT ###############################################
    if ti.static(dim == 3):
        for r in range(n_GPT[None]):
            p = xTilde[GPT[r, 0]] * alpha + x[GPT[r, 0]] * (1 - alpha)
            t0 = xTilde[GPT[r, 1]] * alpha + x[GPT[r, 1]] * (1 - alpha)
            t1 = xTilde[GPT[r, 2]] * alpha + x[GPT[r, 2]] * (1 - alpha)
            t2 = xTilde[GPT[r, 3]] * alpha + x[GPT[r, 3]] * (1 - alpha)
            y_GPT[r, 0], y_GPT[r, 1], y_GPT[r, 2] = p - t0, p - t1, p - t2
            r_GPT[r, 0], r_GPT[r, 1], r_GPT[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)

            p, t0, t1, t2 = x[GPT[r, 0]], x[GPT[r, 1]], x[GPT[r, 2]], x[GPT[r, 3]]
            # Q_GPT[r, 0] = min(max(ti.sqrt(GPT_hessian(ti.Matrix.zero(real, dim), p - t0, p - t1, p - t2, dHat2, kappa).norm()), min_Q), max_Q)
            Q_GPT[r, 0] = ti.sqrt(GPT_hessian(ti.Matrix.zero(real, dim), p - t0, p - t1, p - t2, dHat2, kappa).norm())
    ############################################### EE ###############################################
    if ti.static(dim == 3):
        for r in range(n_GEE[None]):
            a0 = xTilde[GEE[r, 0]] * alpha + x[GEE[r, 0]] * (1 - alpha)
            a1 = xTilde[GEE[r, 1]] * alpha + x[GEE[r, 1]] * (1 - alpha)
            b0 = xTilde[GEE[r, 2]] * alpha + x[GEE[r, 2]] * (1 - alpha)
            b1 = xTilde[GEE[r, 3]] * alpha + x[GEE[r, 3]] * (1 - alpha)
            y_GEE[r, 0], y_GEE[r, 1], y_GEE[r, 2] = a0 - a1, a0 - b0, a0 - b1
            r_GEE[r, 0], r_GEE[r, 1], r_GEE[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)

            a0, a1, b0, b1 = x[GEE[r, 0]], x[GEE[r, 1]], x[GEE[r, 2]], x[GEE[r, 3]]
            # Q_GEE[r, 0] = min(max(ti.sqrt(GEE_hessian(ti.Matrix.zero(real, dim), a0 - a1, a0 - b0, a0 - b1, dHat2, kappa).norm()), min_Q), max_Q)
            Q_GEE[r, 0] = ti.sqrt(GEE_hessian(ti.Matrix.zero(real, dim), a0 - a1, a0 - b0, a0 - b1, dHat2, kappa).norm())
    ############################################### EEM ###############################################
    if ti.static(dim == 3):
        for r in range(n_GEEM[None]):
            a0 = xTilde[GEEM[r, 0]] * alpha + x[GEEM[r, 0]] * (1 - alpha)
            a1 = xTilde[GEEM[r, 1]] * alpha + x[GEEM[r, 1]] * (1 - alpha)
            b0 = xTilde[GEEM[r, 2]] * alpha + x[GEEM[r, 2]] * (1 - alpha)
            b1 = xTilde[GEEM[r, 3]] * alpha + x[GEEM[r, 3]] * (1 - alpha)
            y_GEEM[r, 0], y_GEEM[r, 1], y_GEEM[r, 2] = a0 - a1, a0 - b0, a0 - b1
            r_GEEM[r, 0], r_GEEM[r, 1], r_GEEM[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)

            a0, a1, b0, b1 = x[GEEM[r, 0]], x[GEEM[r, 1]], x[GEEM[r, 2]], x[GEEM[r, 3]]
            _a0, _a1, _b0, _b1 = x0[GEEM[r, 0]], x0[GEEM[r, 1]], x0[GEEM[r, 2]], x0[GEEM[r, 3]]
            # Q_GEEM[r, 0] = min(max(ti.sqrt(GEEM_hessian(ti.Matrix.zero(real, dim), a0 - a1, a0 - b0, a0 - b1, _a0, _a1, _b0, _b1, dHat2, kappa).norm()), min_Q), max_Q)
            Q_GEEM[r, 0] = ti.sqrt(GEEM_hessian(ti.Matrix.zero(real, dim), a0 - a1, a0 - b0, a0 - b1, _a0, _a1, _b0, _b1, dHat2, kappa).norm())
    # reuse y, r
    for c in range(old_n_GPE[None]):
        for d in range(n_GPE[None]):
            if old_GPE[c, 0] == GPE[d, 0] and old_GPE[c, 1] == GPE[d, 1] and old_GPE[c, 2] == GPE[d, 2]:
                k = 1.
                if ti.static(update_dbdf):
                    k = old_Q_GPE[c, 0] / Q_GPE[c, 0]
                else:
                    Q_GPE[d, 0], Q_GPE[d, 1] = old_Q_GPE[c, 0], old_Q_GPE[c, 1]
                y_GPE[d, 0], y_GPE[d, 1] = old_y_GPE[c, 0], old_y_GPE[c, 1]
                r_GPE[d, 0], r_GPE[d, 1] = old_r_GPE[c, 0] * k, old_r_GPE[c, 1] * k
    for c in range(old_n_GPT[None]):
        for d in range(n_GPT[None]):
            if old_GPT[c, 0] == GPT[d, 0] and old_GPT[c, 1] == GPT[d, 1] and old_GPT[c, 2] == GPT[d, 2] and old_GPT[c, 3] == GPT[d, 3]:
                k = 1.
                if ti.static(update_dbdf):
                    k = old_Q_GPT[c, 0] / Q_GPT[c, 0]
                else:
                    Q_GPT[d, 0], Q_GPT[d, 1], Q_GPT[d, 2] = old_Q_GPT[c, 0], old_Q_GPT[c, 1], old_Q_GPT[c, 2]
                y_GPT[d, 0], y_GPT[d, 1], y_GPT[d, 2] = old_y_GPT[c, 0], old_y_GPT[c, 1], old_y_GPT[c, 2]
                r_GPT[d, 0], r_GPT[d, 1], r_GPT[d, 2] = old_r_GPT[c, 0] * k, old_r_GPT[c, 1] * k, old_r_GPT[c, 2] * k
    for c in range(old_n_GEE[None]):
        for d in range(n_GEE[None]):
            if old_GEE[c, 0] == GEE[d, 0] and old_GEE[c, 1] == GEE[d, 1] and old_GEE[c, 2] == GEE[d, 2] and old_GEE[c, 3] == GEE[d, 3]:
                k = 1.
                if ti.static(update_dbdf):
                    k = old_Q_GEE[c, 0] / Q_GEE[c, 0]
                else:
                    Q_GEE[d, 0], Q_GEE[d, 1], Q_GEE[d, 2] = old_Q_GEE[c, 0], old_Q_GEE[c, 1], old_Q_GEE[c, 2]
                y_GEE[d, 0], y_GEE[d, 1], y_GEE[d, 2] = old_y_GEE[c, 0], old_y_GEE[c, 1], old_y_GEE[c, 2]
                r_GEE[d, 0], r_GEE[d, 1], r_GEE[d, 2] = old_r_GEE[c, 0] * k, old_r_GEE[c, 1] * k, old_r_GEE[c, 2] * k
    for c in range(old_n_GEEM[None]):
        for d in range(n_GEEM[None]):
            if old_GEEM[c, 0] == GEEM[d, 0] and old_GEEM[c, 1] == GEEM[d, 1] and old_GEEM[c, 2] == GEEM[d, 2] and old_GEEM[c, 3] == GEEM[d, 3]:
                k = 1.
                if ti.static(update_dbdf):
                    k = old_Q_GEEM[c, 0] / Q_GEEM[c, 0]
                else:
                    Q_GEEM[d, 0], Q_GEEM[d, 1], Q_GEEM[d, 2] = old_Q_GEEM[c, 0], old_Q_GEEM[c, 1], old_Q_GEEM[c, 2]
                y_GEEM[d, 0], y_GEEM[d, 1], y_GEEM[d, 2] = old_y_GEEM[c, 0], old_y_GEEM[c, 1], old_y_GEEM[c, 2]
                r_GEEM[d, 0], r_GEEM[d, 1], r_GEEM[d, 2] = old_r_GEEM[c, 0] * k, old_r_GEEM[c, 1] * k, old_r_GEEM[c, 2] * k


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
                        if dim == 2:
                            global_GPE(_data_row, _data_col, _data_val)
                        elif dim == 3:
                            global_GPT(_data_row, _data_col, _data_val)
                            global_GEE(_data_row, _data_col, _data_val)
                            global_GEEM(_data_row, _data_col, _data_val)
                    with Timer("Global Solve"):
                        solve_system(f * dt)

                    with Timer("Local Step"):
                        local_elasticity()
                        if dim == 2:
                            local_GPE()
                        elif dim == 3:
                            local_GPT()
                            local_GEE()
                            local_GEEM()

                    with Timer("Compute Residual"):
                        print("Prime residual: ", prime_residual(), ", Dual residual: ", dual_residual(), ",Newton residual: ", newton_gradient_residual())

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
