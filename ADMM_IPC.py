from hashlib import sha1
import sys, os, time, math
import taichi as ti
import taichi_three as t3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from fixed_corotated_3d import *
from math_tools import *
from ipc import *
from reader import *

##############################################################################

mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value = read(int(sys.argv[1]))
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

ti.init(arch=ti.cpu)

real = ti.f32
scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

dim = 3
dt = 0.01
E = 1e4
nu = 0.4
la = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
density = 100
n_particles = len(mesh_particles)
n_elements = len(mesh_elements)
n_boundary_points = len(boundary_points_)
n_boundary_edges = len(boundary_edges_)
n_boundary_triangles = len(boundary_triangles_)

x, x0, xx, xTilde, xn, v, m = vec(), vec(), vec(), vec(), vec(), vec(), scalar()
restT = mat()
vertices = ti.var(ti.i32)
W, z, zz, u = scalar(), mat(), mat(), mat()
boundary_points = ti.var(ti.i32)
boundary_edges = ti.var(ti.i32)
boundary_triangles = ti.var(ti.i32)
ti.root.dense(ti.k, n_particles).place(x, x0, xx, xTilde, xn, v, m)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.ij, (n_elements, dim + 1)).place(vertices)
ti.root.dense(ti.i, n_elements).place(W, z, zz, u)
ti.root.dense(ti.i, n_boundary_points).place(boundary_points)
ti.root.dense(ti.ij, (n_boundary_edges, 2)).place(boundary_edges)
ti.root.dense(ti.ij, (n_boundary_triangles, 3)).place(boundary_triangles)

data_rhs = ti.var(real, shape=n_particles * dim)
data_row = ti.var(ti.i32, shape=2000000)
data_col = ti.var(ti.i32, shape=2000000)
data_val = ti.var(real, shape=2000000)
data_x = ti.var(real, shape=n_particles * dim)
cnt = ti.var(dt=ti.i32, shape=())

# n_constraint = 1000000
# constraints = ti.var(ti.i32, shape=(n_constraint, 3))
# old_constraints = ti.var(ti.i32, shape=(n_constraint, 3))
# cc = ti.var(dt=ti.i32, shape=())
# old_cc = ti.var(dt=ti.i32, shape=())
# y, yy, r, old_y, old_r = vec(), vec(), vec(), vec(), vec()
# ti.root.dense(ti.ij, (n_constraint, 2)).place(y, yy, r, old_y, old_r)
# Q, old_Q = scalar(), scalar()
# ti.root.dense(ti.i, n_constraint).place(Q, old_Q)

Q = 1.0

MAX_C = 100000
PP = ti.var(ti.i32, shape=(MAX_C, 2))
n_PP = ti.var(dt=ti.i32, shape=())
y_PP, r_PP = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 1)).place(y_PP, r_PP)
PE = ti.var(ti.i32, shape=(MAX_C, 3))
n_PE = ti.var(dt=ti.i32, shape=())
y_PE, r_PE = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 2)).place(y_PE, r_PE)
PT = ti.var(ti.i32, shape=(MAX_C, 4))
n_PT = ti.var(dt=ti.i32, shape=())
y_PT, r_PT = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_PT, r_PT)
EE = ti.var(ti.i32, shape=(MAX_C, 4))
n_EE = ti.var(dt=ti.i32, shape=())
y_EE, r_EE = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_EE, r_EE)
EEM = ti.var(ti.i32, shape=(MAX_C, 4))
n_EEM = ti.var(dt=ti.i32, shape=())
y_EEM, r_EEM = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_EEM, r_EEM)
PPM = ti.var(ti.i32, shape=(MAX_C, 4))
n_PPM = ti.var(dt=ti.i32, shape=())
y_PPM, r_PPM = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_PPM, r_PPM)
PEM = ti.var(ti.i32, shape=(MAX_C, 4))
n_PEM = ti.var(dt=ti.i32, shape=())
y_PEM, r_PEM = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 3)).place(y_PEM, r_PEM)

old_PP = ti.var(ti.i32, shape=(MAX_C, 2))
old_n_PP = ti.var(dt=ti.i32, shape=())
old_y_PP, old_r_PP = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 1)).place(old_y_PP, old_r_PP)
old_PE = ti.var(ti.i32, shape=(MAX_C, 3))
old_n_PE = ti.var(dt=ti.i32, shape=())
old_y_PE, old_r_PE = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 2)).place(old_y_PE, old_r_PE)
old_PT = ti.var(ti.i32, shape=(MAX_C, 4))
old_n_PT = ti.var(dt=ti.i32, shape=())
old_y_PT, old_r_PT = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_PT, old_r_PT)
old_EE = ti.var(ti.i32, shape=(MAX_C, 4))
old_n_EE = ti.var(dt=ti.i32, shape=())
old_y_EE, old_r_EE = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_EE, old_r_EE)
old_EEM = ti.var(ti.i32, shape=(MAX_C, 4))
old_n_EEM = ti.var(dt=ti.i32, shape=())
old_y_EEM, old_r_EEM = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_EEM, old_r_EEM)
old_PPM = ti.var(ti.i32, shape=(MAX_C, 4))
old_n_PPM = ti.var(dt=ti.i32, shape=())
old_y_PPM, old_r_PPM = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_PPM, old_r_PPM)
old_PEM = ti.var(ti.i32, shape=(MAX_C, 4))
old_n_PEM = ti.var(dt=ti.i32, shape=())
old_y_PEM, old_r_PEM = vec(), vec()
ti.root.dense(ti.ij, (MAX_C, 3)).place(old_y_PEM, old_r_PEM)

dHat2 = 1e-5
dHat = dHat2 ** 0.5
kappa = 1e4


@ti.kernel
def find_constraints():
    old_n_PP[None] = n_PP[None]
    for c in range(old_n_PP[None]):
        old_PP[c, 0], old_PP[c, 1] = PP[c, 0], PP[c, 1]
        old_y_PP[c, 0] = y_PP[c, 0]
        old_r_PP[c, 0] = r_PP[c, 0]
    old_n_PE[None] = n_PE[None]
    for c in range(old_n_PE[None]):
        old_PE[c, 0], old_PE[c, 1], old_PE[c, 2] = PE[c, 0], PE[c, 1], PE[c, 2]
        old_y_PE[c, 0], old_y_PE[c, 1] = y_PE[c, 0], y_PE[c, 1]
        old_r_PE[c, 0], old_r_PE[c, 1] = r_PE[c, 0], r_PE[c, 1]
    old_n_PT[None] = n_PT[None]
    for c in range(old_n_PT[None]):
        old_PT[c, 0], old_PT[c, 1], old_PT[c, 2], old_PT[c, 3] = PT[c, 0], PT[c, 1], PT[c, 2], PT[c, 3]
        old_y_PT[c, 0], old_y_PT[c, 1], old_y_PT[c, 2] = y_PT[c, 0], y_PT[c, 1], y_PT[c, 2]
        old_r_PT[c, 0], old_r_PT[c, 1], old_r_PT[c, 2] = r_PT[c, 0], r_PT[c, 1], r_PT[c, 2]
    old_n_EE[None] = n_EE[None]
    for c in range(old_n_EE[None]):
        old_EE[c, 0], old_EE[c, 1], old_EE[c, 2], old_EE[c, 3] = EE[c, 0], EE[c, 1], EE[c, 2], EE[c, 3]
        old_y_EE[c, 0], old_y_EE[c, 1], old_y_EE[c, 2] = y_EE[c, 0], y_EE[c, 1], y_EE[c, 2]
        old_r_EE[c, 0], old_r_EE[c, 1], old_r_EE[c, 2] = r_EE[c, 0], r_EE[c, 1], r_EE[c, 2]
    old_n_EEM[None] = n_EEM[None]
    for c in range(old_n_EEM[None]):
        old_EEM[c, 0], old_EEM[c, 1], old_EEM[c, 2], old_EEM[c, 3] = EEM[c, 0], EEM[c, 1], EEM[c, 2], EEM[c, 3]
        old_y_EEM[c, 0], old_y_EEM[c, 1], old_y_EEM[c, 2] = y_EEM[c, 0], y_EEM[c, 1], y_EEM[c, 2]
        old_r_EEM[c, 0], old_r_EEM[c, 1], old_r_EEM[c, 2] = r_EEM[c, 0], r_EEM[c, 1], r_EEM[c, 2]
    old_n_PPM[None] = n_PPM[None]
    for c in range(old_n_PPM[None]):
        old_PPM[c, 0], old_PPM[c, 1], old_PPM[c, 2], old_PPM[c, 3] = PPM[c, 0], PPM[c, 1], PPM[c, 2], PPM[c, 3]
        old_y_PPM[c, 0], old_y_PPM[c, 1], old_y_PPM[c, 2] = y_PPM[c, 0], y_PPM[c, 1], y_PPM[c, 2]
        old_r_PPM[c, 0], old_r_PPM[c, 1], old_r_PPM[c, 2] = r_PPM[c, 0], r_PPM[c, 1], r_PPM[c, 2]
    old_n_PEM[None] = n_PEM[None]
    for c in range(old_n_PEM[None]):
        old_PEM[c, 0], old_PEM[c, 1], old_PEM[c, 2], old_PEM[c, 3] = PEM[c, 0], PEM[c, 1], PEM[c, 2], PEM[c, 3]
        old_y_PEM[c, 0], old_y_PEM[c, 1], old_y_PEM[c, 2] = y_PEM[c, 0], y_PEM[c, 1], y_PEM[c, 2]
        old_r_PEM[c, 0], old_r_PEM[c, 1], old_r_PEM[c, 2] = r_PEM[c, 0], r_PEM[c, 1], r_PEM[c, 2]

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
    # xTilde initiated y, r
    for r in range(n_PP[None]):
        p0, p1 = xTilde[PP[r, 0]], xTilde[PP[r, 1]]
        y_PP[r, 0] = p0 - p1
        r_PP[r, 0] = ti.Matrix.zero(real, dim)
    for r in range(n_PE[None]):
        p, e0, e1 = xTilde[PE[r, 0]], xTilde[PE[r, 1]], xTilde[PE[r, 2]]
        y_PE[r, 0], y_PE[r, 1] = p - e0, p - e1
        r_PE[r, 0], r_PE[r, 1] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)
    for r in range(n_PT[None]):
        p, t0, t1, t2 = xTilde[PT[r, 0]], xTilde[PT[r, 1]], xTilde[PT[r, 2]], xTilde[PT[r, 3]]
        y_PT[r, 0], y_PT[r, 1], y_PT[r, 2] = p - t0, p - t1, p - t2
        r_PT[r, 0], r_PT[r, 1], r_PT[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)
    for r in range(n_EE[None]):
        a0, a1, b0, b1 = xTilde[EE[r, 0]], xTilde[EE[r, 1]], xTilde[EE[r, 2]], xTilde[EE[r, 3]]
        y_EE[r, 0], y_EE[r, 1], y_EE[r, 2] = a0 - a1, a0 - b0, a0 - b1
        r_EE[r, 0], r_EE[r, 1], r_EE[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)
    for r in range(n_EEM[None]):
        a0, a1, b0, b1 = xTilde[EEM[r, 0]], xTilde[EEM[r, 1]], xTilde[EEM[r, 2]], xTilde[EEM[r, 3]]
        y_EEM[r, 0], y_EEM[r, 1], y_EEM[r, 2] = a0 - a1, a0 - b0, a0 - b1
        r_EEM[r, 0], r_EEM[r, 1], r_EEM[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)
    for r in range(n_PPM[None]):
        a0, a1, b0, b1 = xTilde[PPM[r, 0]], xTilde[PPM[r, 1]], xTilde[PPM[r, 2]], xTilde[PPM[r, 3]]
        y_PPM[r, 0], y_PPM[r, 1], y_PPM[r, 2] = a0 - a1, a0 - b0, a0 - b1
        r_PPM[r, 0], r_PPM[r, 1], r_PPM[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)
    for r in range(n_PEM[None]):
        a0, a1, b0, b1 = xTilde[PEM[r, 0]], xTilde[PEM[r, 1]], xTilde[PEM[r, 2]], xTilde[PEM[r, 3]]
        y_PEM[r, 0], y_PEM[r, 1], y_PEM[r, 2] = a0 - a1, a0 - b0, a0 - b1
        r_PEM[r, 0], r_PEM[r, 1], r_PEM[r, 2] = ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim), ti.Matrix.zero(real, dim)
    # reuse y, r
    for c in range(old_n_PP[None]):
        for d in range(n_PP[None]):
            if old_PP[c, 0] == PP[d, 0] and old_PP[c, 1] == PP[d, 1]:
                y_PP[d, 0] = old_y_PP[c, 0]
                r_PP[d, 0] = old_r_PP[c, 0]
    for c in range(old_n_PE[None]):
        for d in range(n_PE[None]):
            if old_PE[c, 0] == PE[d, 0] and old_PE[c, 1] == PE[d, 1] and old_PE[c, 2] == PE[d, 2]:
                y_PE[d, 0], y_PE[d, 1] = old_y_PE[c, 0], old_y_PE[c, 1]
                r_PE[d, 0], r_PE[d, 1] = old_r_PE[c, 0], old_r_PE[c, 1]
    for c in range(old_n_PT[None]):
        for d in range(n_PT[None]):
            if old_PT[c, 0] == PT[d, 0] and old_PT[c, 1] == PT[d, 1] and old_PT[c, 2] == PT[d, 2] and old_PT[c, 3] == PT[d, 3]:
                y_PT[d, 0], y_PT[d, 1], y_PT[d, 2] = old_y_PT[c, 0], old_y_PT[c, 1], old_y_PT[c, 2]
                r_PT[d, 0], r_PT[d, 1], r_PT[d, 2] = old_r_PT[c, 0], old_r_PT[c, 1], old_r_PT[c, 2]
    for c in range(old_n_EE[None]):
        for d in range(n_EE[None]):
            if old_EE[c, 0] == EE[d, 0] and old_EE[c, 1] == EE[d, 1] and old_EE[c, 2] == EE[d, 2] and old_EE[c, 3] == EE[d, 3]:
                y_EE[d, 0], y_EE[d, 1], y_EE[d, 2] = old_y_EE[c, 0], old_y_EE[c, 1], old_y_EE[c, 2]
                r_EE[d, 0], r_EE[d, 1], r_EE[d, 2] = old_r_EE[c, 0], old_r_EE[c, 1], old_r_EE[c, 2]
    for c in range(old_n_EEM[None]):
        for d in range(n_EEM[None]):
            if old_EEM[c, 0] == EEM[d, 0] and old_EEM[c, 1] == EEM[d, 1] and old_EEM[c, 2] == EEM[d, 2] and old_EEM[c, 3] == EEM[d, 3]:
                y_EEM[d, 0], y_EEM[d, 1], y_EEM[d, 2] = old_y_EEM[c, 0], old_y_EEM[c, 1], old_y_EEM[c, 2]
                r_EEM[d, 0], r_EEM[d, 1], r_EEM[d, 2] = old_r_EEM[c, 0], old_r_EEM[c, 1], old_r_EEM[c, 2]
    for c in range(old_n_PPM[None]):
        for d in range(n_PPM[None]):
            if old_PPM[c, 0] == PPM[d, 0] and old_PPM[c, 1] == PPM[d, 1] and old_PPM[c, 2] == PPM[d, 2] and old_PPM[c, 3] == PPM[d, 3]:
                y_PPM[d, 0], y_PPM[d, 1], y_PPM[d, 2] = old_y_PPM[c, 0], old_y_PPM[c, 1], old_y_PPM[c, 2]
                r_PPM[d, 0], r_PPM[d, 1], r_PPM[d, 2] = old_r_PPM[c, 0], old_r_PPM[c, 1], old_r_PPM[c, 2]
    for c in range(old_n_PEM[None]):
        for d in range(n_PEM[None]):
            if old_PEM[c, 0] == PEM[d, 0] and old_PEM[c, 1] == PEM[d, 1] and old_PEM[c, 2] == PEM[d, 2] and old_PEM[c, 3] == PEM[d, 3]:
                y_PEM[d, 0], y_PEM[d, 1], y_PEM[d, 2] = old_y_PEM[c, 0], old_y_PEM[c, 1], old_y_PEM[c, 2]
                r_PEM[d, 0], r_PEM[d, 1], r_PEM[d, 2] = old_r_PEM[c, 0], old_r_PEM[c, 1], old_r_PEM[c, 2]


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
    for _ in range(1):
        for i in range(n_elements):
            restT[i] = compute_T(i)
            mass = restT[i].determinant() / dim / (dim - 1) * density / (dim + 1)
            if mass < 0.0:
                print("FATAL ERROR : mesh inverted")
            for d in ti.static(range(dim + 1)):
                m[vertices[i, d]] += mass
    for i in range(n_particles):
        x0[i] = x[i]
        v(0)[i] = 2.0 if x(0)[i] < 0 else -2.0


@ti.kernel
def initial_guess():
    # set W, u, z
    for i in range(n_elements):
        currentT = compute_T(i)
        W[i] = ti.sqrt(la + mu * 2 / 3) * (restT[i].determinant() / dim / (dim - 1))
        z[i] = currentT @ restT[i].inverse()
        u[i] = ti.Matrix.zero(real, dim, dim)
    for i in range(n_particles):
        xn[i] = x[i]
        xTilde[i] = x[i] + dt * v[i]
        # xTilde(1)[i] -= dt * dt * 9.8


@ti.func
def X2F(p: ti.template(), q: ti.template(), i: ti.template(), j: ti.template(), A):
    val = 0.0
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
def global_step():
    cnt[None] = 0
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            c = i * dim + d
            data_row[c] = i * dim + d
            data_col[c] = i * dim + d
            data_val[c] = m[i]
            data_rhs[i * dim + d] += m[i] * xTilde(d)[i]
    cnt[None] += n_particles * dim
    for _ in range(1):
        for e in range(n_elements):
            A = restT[e].inverse()
            for p in ti.static(range(dim + 1)):
                for i in ti.static(range(dim)):
                    for j in ti.static(range(dim)):
                        for pp in ti.static(range(dim + 1)):
                            q, qq = i, i
                            c = cnt[None] + e * dim * dim * (dim + 1) * (dim + 1) + p * dim * dim * (dim + 1) + i * dim * (dim + 1) + j * (dim + 1) + pp
                            data_row[c] = vertices[e, p] * dim + q
                            data_col[c] = vertices[e, pp] * dim + qq
                            data_val[c] = X2F(p, q, i, j, A) * X2F(pp, qq, i, j, A) * W[e] * W[e]
            F = z[e] - u[e]
            for p in ti.static(range(dim + 1)):
                for i in ti.static(range(dim)):
                    for j in ti.static(range(dim)):
                        q = i
                        data_rhs[vertices[e, p] * dim + q] += X2F(p, q, i, j, A) * F[i, j] * W[e] * W[e]
    cnt[None] += n_elements * dim * dim * (dim + 1) * (dim + 1)
    # ETE = ti.Matrix([[2, -1, -1], [-1, 1, 0], [-1, 0, 1]])
    # for _ in range(1):
    #     for c in range(cc[None]):
    #         for p in ti.static(range(3)):
    #             for q in ti.static(range(3)):
    #                 for j in ti.static(range(2)):
    #                     idx = cnt[None] + c * 18 + p * 6 + q * 2 + j
    #                     data_mat[0, idx] = constraints[c, p] * 2 + j
    #                     data_mat[1, idx] = constraints[c, q] * 2 + j
    #                     data_mat[2, idx] = ETE[p, q] * Q[c] * Q[c]
    #         for j in ti.static(range(2)):
    #             data_rhs[constraints[c, 0] * 2 + j] += (y(j)[c, 0] - r(j)[c, 0]) * Q[c] * Q[c]
    #             data_rhs[constraints[c, 0] * 2 + j] += (y(j)[c, 1] - r(j)[c, 1]) * Q[c] * Q[c]
    #             data_rhs[constraints[c, 1] * 2 + j] -= (y(j)[c, 0] - r(j)[c, 0]) * Q[c] * Q[c]
    #             data_rhs[constraints[c, 2] * 2 + j] -= (y(j)[c, 1] - r(j)[c, 1]) * Q[c] * Q[c]
    # cnt[None] += cc[None] * 18
@ti.kernel
def global_PP():
    ETE2 = ti.Matrix([[1, -1], [-1, 1]])
    for c in range(n_PP[None]):
        for p in ti.static(range(2)):
            for q in ti.static(range(2)):
                for j in ti.static(range(3)):
                    idx = cnt[None] + c * 12 + p * 6 + q * 3 + j
                    data_row[idx] = PP[c, p] * 3 + j
                    data_col[idx] = PP[c, q] * 3 + j
                    data_val[idx] = ETE2[p, q] * Q * Q
        for j in ti.static(range(3)):
            data_rhs[PP[c, 0] * 3 + j] += (y_PP(j)[c, 0] - r_PP(j)[c, 0]) * Q * Q
            data_rhs[PP[c, 1] * 3 + j] -= (y_PP(j)[c, 0] - r_PP(j)[c, 0]) * Q * Q
    cnt[None] += n_PP[None] * 12
@ti.kernel
def global_PE():
    ETE3 = ti.Matrix([[2, -1, -1], [-1, 1, 0], [-1, 0, 1]])
    for c in range(n_PE[None]):
        for p in ti.static(range(3)):
            for q in ti.static(range(3)):
                for j in ti.static(range(3)):
                    idx = cnt[None] + c * 27 + p * 9 + q * 3 + j
                    data_row[idx] = PE[c, p] * 3 + j
                    data_col[idx] = PE[c, q] * 3 + j
                    data_val[idx] = ETE3[p, q] * Q * Q
        for j in ti.static(range(3)):
            data_rhs[PE[c, 0] * 3 + j] += (y_PE(j)[c, 0] - r_PE(j)[c, 0]) * Q * Q
            data_rhs[PE[c, 0] * 3 + j] += (y_PE(j)[c, 1] - r_PE(j)[c, 1]) * Q * Q
            data_rhs[PE[c, 1] * 3 + j] -= (y_PE(j)[c, 0] - r_PE(j)[c, 0]) * Q * Q
            data_rhs[PE[c, 2] * 3 + j] -= (y_PE(j)[c, 1] - r_PE(j)[c, 1]) * Q * Q
    cnt[None] += n_PE[None] * 27
@ti.kernel
def global_PT():
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for c in range(n_PT[None]):
        for p in ti.static(range(4)):
            for q in ti.static(range(4)):
                for j in ti.static(range(3)):
                    idx = cnt[None] + c * 48 + p * 12 + q * 3 + j
                    data_row[idx] = PT[c, p] * 3 + j
                    data_col[idx] = PT[c, q] * 3 + j
                    data_val[idx] = ETE4[p, q] * Q * Q
        for j in ti.static(range(3)):
            data_rhs[PT[c, 0] * 3 + j] += (y_PT(j)[c, 0] - r_PT(j)[c, 0]) * Q * Q
            data_rhs[PT[c, 0] * 3 + j] += (y_PT(j)[c, 1] - r_PT(j)[c, 1]) * Q * Q
            data_rhs[PT[c, 0] * 3 + j] += (y_PT(j)[c, 2] - r_PT(j)[c, 2]) * Q * Q
            data_rhs[PT[c, 1] * 3 + j] -= (y_PT(j)[c, 0] - r_PT(j)[c, 0]) * Q * Q
            data_rhs[PT[c, 2] * 3 + j] -= (y_PT(j)[c, 1] - r_PT(j)[c, 1]) * Q * Q
            data_rhs[PT[c, 3] * 3 + j] -= (y_PT(j)[c, 2] - r_PT(j)[c, 2]) * Q * Q
    cnt[None] += n_PT[None] * 48
@ti.kernel
def global_EE():
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for c in range(n_EE[None]):
        for p in ti.static(range(4)):
            for q in ti.static(range(4)):
                for j in ti.static(range(3)):
                    idx = cnt[None] + c * 48 + p * 12 + q * 3 + j
                    data_row[idx] = EE[c, p] * 3 + j
                    data_col[idx] = EE[c, q] * 3 + j
                    data_val[idx] = ETE4[p, q] * Q * Q
        for j in ti.static(range(3)):
            data_rhs[EE[c, 0] * 3 + j] += (y_EE(j)[c, 0] - r_EE(j)[c, 0]) * Q * Q
            data_rhs[EE[c, 0] * 3 + j] += (y_EE(j)[c, 1] - r_EE(j)[c, 1]) * Q * Q
            data_rhs[EE[c, 0] * 3 + j] += (y_EE(j)[c, 2] - r_EE(j)[c, 2]) * Q * Q
            data_rhs[EE[c, 1] * 3 + j] -= (y_EE(j)[c, 0] - r_EE(j)[c, 0]) * Q * Q
            data_rhs[EE[c, 2] * 3 + j] -= (y_EE(j)[c, 1] - r_EE(j)[c, 1]) * Q * Q
            data_rhs[EE[c, 3] * 3 + j] -= (y_EE(j)[c, 2] - r_EE(j)[c, 2]) * Q * Q
    cnt[None] += n_EE[None] * 48
@ti.kernel
def global_EEM():
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for c in range(n_EEM[None]):
        for p in ti.static(range(4)):
            for q in ti.static(range(4)):
                for j in ti.static(range(3)):
                    idx = cnt[None] + c * 48 + p * 12 + q * 3 + j
                    data_row[idx] = EEM[c, p] * 3 + j
                    data_col[idx] = EEM[c, q] * 3 + j
                    data_val[idx] = ETE4[p, q] * Q * Q
        for j in ti.static(range(3)):
            data_rhs[EEM[c, 0] * 3 + j] += (y_EEM(j)[c, 0] - r_EEM(j)[c, 0]) * Q * Q
            data_rhs[EEM[c, 0] * 3 + j] += (y_EEM(j)[c, 1] - r_EEM(j)[c, 1]) * Q * Q
            data_rhs[EEM[c, 0] * 3 + j] += (y_EEM(j)[c, 2] - r_EEM(j)[c, 2]) * Q * Q
            data_rhs[EEM[c, 1] * 3 + j] -= (y_EEM(j)[c, 0] - r_EEM(j)[c, 0]) * Q * Q
            data_rhs[EEM[c, 2] * 3 + j] -= (y_EEM(j)[c, 1] - r_EEM(j)[c, 1]) * Q * Q
            data_rhs[EEM[c, 3] * 3 + j] -= (y_EEM(j)[c, 2] - r_EEM(j)[c, 2]) * Q * Q
    cnt[None] += n_EEM[None] * 48
@ti.kernel
def global_PPM():
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for c in range(n_PPM[None]):
        for p in ti.static(range(4)):
            for q in ti.static(range(4)):
                for j in ti.static(range(3)):
                    idx = cnt[None] + c * 48 + p * 12 + q * 3 + j
                    data_row[idx] = PPM[c, p] * 3 + j
                    data_col[idx] = PPM[c, q] * 3 + j
                    data_val[idx] = ETE4[p, q] * Q * Q
        for j in ti.static(range(3)):
            data_rhs[PPM[c, 0] * 3 + j] += (y_PPM(j)[c, 0] - r_PPM(j)[c, 0]) * Q * Q
            data_rhs[PPM[c, 0] * 3 + j] += (y_PPM(j)[c, 1] - r_PPM(j)[c, 1]) * Q * Q
            data_rhs[PPM[c, 0] * 3 + j] += (y_PPM(j)[c, 2] - r_PPM(j)[c, 2]) * Q * Q
            data_rhs[PPM[c, 1] * 3 + j] -= (y_PPM(j)[c, 0] - r_PPM(j)[c, 0]) * Q * Q
            data_rhs[PPM[c, 2] * 3 + j] -= (y_PPM(j)[c, 1] - r_PPM(j)[c, 1]) * Q * Q
            data_rhs[PPM[c, 3] * 3 + j] -= (y_PPM(j)[c, 2] - r_PPM(j)[c, 2]) * Q * Q
    cnt[None] += n_PPM[None] * 48
@ti.kernel
def global_PEM():
    ETE4 = ti.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    for c in range(n_PEM[None]):
        for p in ti.static(range(4)):
            for q in ti.static(range(4)):
                for j in ti.static(range(3)):
                    idx = cnt[None] + c * 48 + p * 12 + q * 3 + j
                    data_row[idx] = PEM[c, p] * 3 + j
                    data_col[idx] = PEM[c, q] * 3 + j
                    data_val[idx] = ETE4[p, q] * Q * Q
        for j in ti.static(range(3)):
            data_rhs[PEM[c, 0] * 3 + j] += (y_PEM(j)[c, 0] - r_PEM(j)[c, 0]) * Q * Q
            data_rhs[PEM[c, 0] * 3 + j] += (y_PEM(j)[c, 1] - r_PEM(j)[c, 1]) * Q * Q
            data_rhs[PEM[c, 0] * 3 + j] += (y_PEM(j)[c, 2] - r_PEM(j)[c, 2]) * Q * Q
            data_rhs[PEM[c, 1] * 3 + j] -= (y_PEM(j)[c, 0] - r_PEM(j)[c, 0]) * Q * Q
            data_rhs[PEM[c, 2] * 3 + j] -= (y_PEM(j)[c, 1] - r_PEM(j)[c, 1]) * Q * Q
            data_rhs[PEM[c, 3] * 3 + j] -= (y_PEM(j)[c, 2] - r_PEM(j)[c, 2]) * Q * Q
    cnt[None] += n_PEM[None] * 48


def solve_system():
    print("Total entries: ", cnt[None])
    row, col, val = data_row.to_numpy()[:cnt[None]], data_col.to_numpy()[:cnt[None]], data_val.to_numpy()[:cnt[None]]
    rhs = data_rhs.to_numpy()
    n = n_particles * dim
    A = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n))
    data_x.from_numpy(scipy.sparse.linalg.spsolve(A, rhs))
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            x(d)[i] = data_x[i * dim + d]


@ti.func
def local_energy(sigma, sigma_Dx_plus_u, vol0, W):
    sig = ti.Matrix([[sigma[0], 0.0, 0.0], [0.0, sigma[1], 0.0], [0.0, 0.0, sigma[2]]])
    return fixed_corotated_energy(sig, la, mu) * dt * dt * vol0 + (sigma - sigma_Dx_plus_u).norm_sqr() * W * W / 2
@ti.func
def local_gradient(sigma, sigma_Dx_plus_u, vol0, W):
    sig = ti.Matrix([[sigma[0], 0.0, 0.0], [0.0, sigma[1], 0.0], [0.0, 0.0, sigma[2]]])
    return fixed_corotated_gradient(sig, la, mu) * dt * dt * vol0 + (sigma - sigma_Dx_plus_u) * W * W
@ti.func
def local_hessian(sigma, sigma_Dx_plus_u, vol0, W):
    sig = ti.Matrix([[sigma[0], 0.0, 0.0], [0.0, sigma[1], 0.0], [0.0, 0.0, sigma[2]]])
    return project_pd_3(fixed_corotated_hessian(sig, la, mu)) * dt * dt * vol0 + ti.Matrix.identity(real, 3) * W * W


@ti.func
def PP_energy(pos, posTilde, Q):
    p0 = ti.Vector([0.0, 0.0, 0.0])
    p1 = ti.Vector([pos[0], pos[1], pos[2]])
    dist2 = PP_3D_E(p0, p1)
    return barrier_E(dist2, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
@ti.func
def PP_gradient(pos, posTilde, Q):
    p0 = ti.Vector([0.0, 0.0, 0.0])
    p1 = ti.Vector([pos[0], pos[1], pos[2]])
    dist2 = PP_3D_E(p0, p1)
    dist2g = PP_3D_g(p0, p1)
    bg = barrier_g(dist2, dHat2, kappa)
    g = bg * dist2g
    return ti.Vector([g[3], g[4], g[5]]) + (pos - posTilde) * Q * Q
@ti.func
def PP_hessian(pos, posTilde, Q):
    p0 = ti.Vector([0.0, 0.0, 0.0])
    p1 = ti.Vector([pos[0], pos[1], pos[2]])
    dist2 = PP_3D_E(p0, p1)
    dist2g = PP_3D_g(p0, p1)
    bg = barrier_g(dist2, dHat2, kappa)
    H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_3D_H(p0, p1)
    eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5]], [H[4, 3], H[4, 4], H[4, 5]], [H[5, 3], H[5, 4], H[5, 5]]])
    return project_pd_3(eH) + ti.Matrix.identity(real, 3) * Q * Q
@ti.func
def PE_energy(pos, posTilde, Q):
    p = ti.Vector([0.0, 0.0, 0.0])
    e0 = ti.Vector([pos[0], pos[1], pos[2]])
    e1 = ti.Vector([pos[3], pos[4], pos[5]])
    dist2 = PE_3D_E(p, e0, e1)
    return barrier_E(dist2, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
@ti.func
def PE_gradient(pos, posTilde, Q):
    p = ti.Vector([0.0, 0.0, 0.0])
    e0 = ti.Vector([pos[0], pos[1], pos[2]])
    e1 = ti.Vector([pos[3], pos[4], pos[5]])
    dist2 = PE_3D_E(p, e0, e1)
    dist2g = PE_3D_g(p, e0, e1)
    bg = barrier_g(dist2, dHat2, kappa)
    g = bg * dist2g
    return ti.Vector([g[3], g[4], g[5], g[6], g[7], g[8]]) + (pos - posTilde) * Q * Q
@ti.func
def PE_hessian(pos, posTilde, Q):
    p = ti.Vector([0.0, 0.0, 0.0])
    e0 = ti.Vector([pos[0], pos[1], pos[2]])
    e1 = ti.Vector([pos[3], pos[4], pos[5]])
    dist2 = PE_3D_E(p, e0, e1)
    dist2g = PE_3D_g(p, e0, e1)
    bg = barrier_g(dist2, dHat2, kappa)
    H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_3D_H(p, e0, e1)
    eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5], H[3, 6], H[3, 7], H[3, 8]], [H[4, 3], H[4, 4], H[4, 5], H[4, 6], H[4, 7], H[4, 8]], [H[5, 3], H[5, 4], H[5, 5], H[5, 6], H[5, 7], H[5, 8]], [H[6, 3], H[6, 4], H[6, 5], H[6, 6], H[6, 7], H[6, 8]], [H[7, 3], H[7, 4], H[7, 5], H[7, 6], H[7, 7], H[7, 8]], [H[8, 3], H[8, 4], H[8, 5], H[8, 6], H[8, 7], H[8, 8]]])
    return project_pd_6(eH) + ti.Matrix.identity(real, 6) * Q * Q
@ti.func
def PT_energy(pos, posTilde, Q):
    p = ti.Vector([0.0, 0.0, 0.0])
    t0 = ti.Vector([pos[0], pos[1], pos[2]])
    t1 = ti.Vector([pos[3], pos[4], pos[5]])
    t2 = ti.Vector([pos[6], pos[7], pos[8]])
    dist2 = PT_3D_E(p, t0, t1, t2)
    return barrier_E(dist2, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
@ti.func
def PT_gradient(pos, posTilde, Q):
    p = ti.Vector([0.0, 0.0, 0.0])
    t0 = ti.Vector([pos[0], pos[1], pos[2]])
    t1 = ti.Vector([pos[3], pos[4], pos[5]])
    t2 = ti.Vector([pos[6], pos[7], pos[8]])
    dist2 = PT_3D_E(p, t0, t1, t2)
    dist2g = PT_3D_g(p, t0, t1, t2)
    bg = barrier_g(dist2, dHat2, kappa)
    g = bg * dist2g
    return ti.Vector([g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11]]) + (pos - posTilde) * Q * Q
@ti.func
def PT_hessian(pos, posTilde, Q):
    p = ti.Vector([0.0, 0.0, 0.0])
    t0 = ti.Vector([pos[0], pos[1], pos[2]])
    t1 = ti.Vector([pos[3], pos[4], pos[5]])
    t2 = ti.Vector([pos[6], pos[7], pos[8]])
    dist2 = PT_3D_E(p, t0, t1, t2)
    dist2g = PT_3D_g(p, t0, t1, t2)
    bg = barrier_g(dist2, dHat2, kappa)
    H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PT_3D_H(p, t0, t1, t2)
    eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5], H[3, 6], H[3, 7], H[3, 8], H[3, 9], H[3, 10], H[3, 11]], [H[4, 3], H[4, 4], H[4, 5], H[4, 6], H[4, 7], H[4, 8], H[4, 9], H[4, 10], H[4, 11]], [H[5, 3], H[5, 4], H[5, 5], H[5, 6], H[5, 7], H[5, 8], H[5, 9], H[5, 10], H[5, 11]], [H[6, 3], H[6, 4], H[6, 5], H[6, 6], H[6, 7], H[6, 8], H[6, 9], H[6, 10], H[6, 11]], [H[7, 3], H[7, 4], H[7, 5], H[7, 6], H[7, 7], H[7, 8], H[7, 9], H[7, 10], H[7, 11]], [H[8, 3], H[8, 4], H[8, 5], H[8, 6], H[8, 7], H[8, 8], H[8, 9], H[8, 10], H[8, 11]], [H[9, 3], H[9, 4], H[9, 5], H[9, 6], H[9, 7], H[9, 8], H[9, 9], H[9, 10], H[9, 11]], [H[10, 3], H[10, 4], H[10, 5], H[10, 6], H[10, 7], H[10, 8], H[10, 9], H[10, 10], H[10, 11]], [H[11, 3], H[11, 4], H[11, 5], H[11, 6], H[11, 7], H[11, 8], H[11, 9], H[11, 10], H[11, 11]]])
    return project_pd_9(eH) + ti.Matrix.identity(real, 9) * Q * Q
@ti.func
def EE_energy(pos, posTilde, Q):
    a0 = ti.Vector([0.0, 0.0, 0.0])
    a1 = ti.Vector([pos[0], pos[1], pos[2]])
    b0 = ti.Vector([pos[3], pos[4], pos[5]])
    b1 = ti.Vector([pos[6], pos[7], pos[8]])
    dist2 = EE_3D_E(a0, a1, b0, b1)
    return barrier_E(dist2, dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2
@ti.func
def EE_gradient(pos, posTilde, Q):
    a0 = ti.Vector([0.0, 0.0, 0.0])
    a1 = ti.Vector([pos[0], pos[1], pos[2]])
    b0 = ti.Vector([pos[3], pos[4], pos[5]])
    b1 = ti.Vector([pos[6], pos[7], pos[8]])
    dist2 = EE_3D_E(a0, a1, b0, b1)
    dist2g = EE_3D_g(a0, a1, b0, b1)
    bg = barrier_g(dist2, dHat2, kappa)
    g = bg * dist2g
    return ti.Vector([g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11]]) + (pos - posTilde) * Q * Q
@ti.func
def EE_hessian(pos, posTilde, Q):
    a0 = ti.Vector([0.0, 0.0, 0.0])
    a1 = ti.Vector([pos[0], pos[1], pos[2]])
    b0 = ti.Vector([pos[3], pos[4], pos[5]])
    b1 = ti.Vector([pos[6], pos[7], pos[8]])
    dist2 = EE_3D_E(a0, a1, b0, b1)
    dist2g = EE_3D_g(a0, a1, b0, b1)
    bg = barrier_g(dist2, dHat2, kappa)
    H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * EE_3D_H(a0, a1, b0, b1)
    eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5], H[3, 6], H[3, 7], H[3, 8], H[3, 9], H[3, 10], H[3, 11]], [H[4, 3], H[4, 4], H[4, 5], H[4, 6], H[4, 7], H[4, 8], H[4, 9], H[4, 10], H[4, 11]], [H[5, 3], H[5, 4], H[5, 5], H[5, 6], H[5, 7], H[5, 8], H[5, 9], H[5, 10], H[5, 11]], [H[6, 3], H[6, 4], H[6, 5], H[6, 6], H[6, 7], H[6, 8], H[6, 9], H[6, 10], H[6, 11]], [H[7, 3], H[7, 4], H[7, 5], H[7, 6], H[7, 7], H[7, 8], H[7, 9], H[7, 10], H[7, 11]], [H[8, 3], H[8, 4], H[8, 5], H[8, 6], H[8, 7], H[8, 8], H[8, 9], H[8, 10], H[8, 11]], [H[9, 3], H[9, 4], H[9, 5], H[9, 6], H[9, 7], H[9, 8], H[9, 9], H[9, 10], H[9, 11]], [H[10, 3], H[10, 4], H[10, 5], H[10, 6], H[10, 7], H[10, 8], H[10, 9], H[10, 10], H[10, 11]], [H[11, 3], H[11, 4], H[11, 5], H[11, 6], H[11, 7], H[11, 8], H[11, 9], H[11, 10], H[11, 11]]])
    return project_pd_9(eH) + ti.Matrix.identity(real, 9) * Q * Q
@ti.func
def EEM_energy(pos, posTilde, Q, r):
    a0 = ti.Vector([0.0, 0.0, 0.0])
    a1 = ti.Vector([pos[0], pos[1], pos[2]])
    b0 = ti.Vector([pos[3], pos[4], pos[5]])
    b1 = ti.Vector([pos[6], pos[7], pos[8]])
    _a0, _a1, _b0, _b1 = x0[EEM[r, 0]], x0[EEM[r, 1]], x0[EEM[r, 2]], x0[EEM[r, 3]]
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = EE_3D_E(a0, a1, b0, b1)
    return barrier_E(dist2, dHat2, kappa) * M_E(a0, a1, b0, b1, eps_x) + (pos - posTilde).norm_sqr() * Q * Q / 2
@ti.func
def EEM_gradient(pos, posTilde, Q, r):
    a0 = ti.Vector([0.0, 0.0, 0.0])
    a1 = ti.Vector([pos[0], pos[1], pos[2]])
    b0 = ti.Vector([pos[3], pos[4], pos[5]])
    b1 = ti.Vector([pos[6], pos[7], pos[8]])
    _a0, _a1, _b0, _b1 = x0[EEM[r, 0]], x0[EEM[r, 1]], x0[EEM[r, 2]], x0[EEM[r, 3]]
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = EE_3D_E(a0, a1, b0, b1)
    dist2g = EE_3D_g(a0, a1, b0, b1)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    lg = bg * dist2g
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    g = lg * M + b * Mg
    return ti.Vector([g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11]]) + (pos - posTilde) * Q * Q
@ti.func
def EEM_hessian(pos, posTilde, Q, r):
    a0 = ti.Vector([0.0, 0.0, 0.0])
    a1 = ti.Vector([pos[0], pos[1], pos[2]])
    b0 = ti.Vector([pos[3], pos[4], pos[5]])
    b1 = ti.Vector([pos[6], pos[7], pos[8]])
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
    H = lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
    eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5], H[3, 6], H[3, 7], H[3, 8], H[3, 9], H[3, 10], H[3, 11]], [H[4, 3], H[4, 4], H[4, 5], H[4, 6], H[4, 7], H[4, 8], H[4, 9], H[4, 10], H[4, 11]], [H[5, 3], H[5, 4], H[5, 5], H[5, 6], H[5, 7], H[5, 8], H[5, 9], H[5, 10], H[5, 11]], [H[6, 3], H[6, 4], H[6, 5], H[6, 6], H[6, 7], H[6, 8], H[6, 9], H[6, 10], H[6, 11]], [H[7, 3], H[7, 4], H[7, 5], H[7, 6], H[7, 7], H[7, 8], H[7, 9], H[7, 10], H[7, 11]], [H[8, 3], H[8, 4], H[8, 5], H[8, 6], H[8, 7], H[8, 8], H[8, 9], H[8, 10], H[8, 11]], [H[9, 3], H[9, 4], H[9, 5], H[9, 6], H[9, 7], H[9, 8], H[9, 9], H[9, 10], H[9, 11]], [H[10, 3], H[10, 4], H[10, 5], H[10, 6], H[10, 7], H[10, 8], H[10, 9], H[10, 10], H[10, 11]], [H[11, 3], H[11, 4], H[11, 5], H[11, 6], H[11, 7], H[11, 8], H[11, 9], H[11, 10], H[11, 11]]])
    return project_pd_9(eH) + ti.Matrix.identity(real, 9) * Q * Q
@ti.func
def PPM_energy(pos, posTilde, Q, r):
    a0 = ti.Vector([0.0, 0.0, 0.0])
    a1 = ti.Vector([pos[0], pos[1], pos[2]])
    b0 = ti.Vector([pos[3], pos[4], pos[5]])
    b1 = ti.Vector([pos[6], pos[7], pos[8]])
    _a0, _a1, _b0, _b1 = x0[PPM[r, 0]], x0[PPM[r, 1]], x0[PPM[r, 2]], x0[PPM[r, 3]]
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PP_3D_E(a0, b0)
    return barrier_E(dist2, dHat2, kappa) * M_E(a0, a1, b0, b1, eps_x) + (pos - posTilde).norm_sqr() * Q * Q / 2
@ti.func
def PPM_gradient(pos, posTilde, Q, r):
    a0 = ti.Vector([0.0, 0.0, 0.0])
    a1 = ti.Vector([pos[0], pos[1], pos[2]])
    b0 = ti.Vector([pos[3], pos[4], pos[5]])
    b1 = ti.Vector([pos[6], pos[7], pos[8]])
    _a0, _a1, _b0, _b1 = x0[PPM[r, 0]], x0[PPM[r, 1]], x0[PPM[r, 2]], x0[PPM[r, 3]]
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PP_3D_E(a0, b0)
    dist2g = PP_3D_g(a0, b0)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    idx = ti.static([0, 1, 2, 6, 7, 8])
    lg = fill_vec(bg * dist2g, idx, 6)
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    g = lg * M + b * Mg
    return ti.Vector([g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11]]) + (pos - posTilde) * Q * Q
@ti.func
def PPM_hessian(pos, posTilde, Q, r):
    a0 = ti.Vector([0.0, 0.0, 0.0])
    a1 = ti.Vector([pos[0], pos[1], pos[2]])
    b0 = ti.Vector([pos[3], pos[4], pos[5]])
    b1 = ti.Vector([pos[6], pos[7], pos[8]])
    _a0, _a1, _b0, _b1 = x0[PPM[r, 0]], x0[PPM[r, 1]], x0[PPM[r, 2]], x0[PPM[r, 3]]
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PP_3D_E(a0, b0)
    dist2g = PP_3D_g(a0, b0)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    idx = ti.static([0, 1, 2, 6, 7, 8])
    lg = fill_vec(bg * dist2g, idx, 6)
    lH = fill_mat(barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_3D_H(a0, b0), idx, 6)
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    H = lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
    eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5], H[3, 6], H[3, 7], H[3, 8], H[3, 9], H[3, 10], H[3, 11]], [H[4, 3], H[4, 4], H[4, 5], H[4, 6], H[4, 7], H[4, 8], H[4, 9], H[4, 10], H[4, 11]], [H[5, 3], H[5, 4], H[5, 5], H[5, 6], H[5, 7], H[5, 8], H[5, 9], H[5, 10], H[5, 11]], [H[6, 3], H[6, 4], H[6, 5], H[6, 6], H[6, 7], H[6, 8], H[6, 9], H[6, 10], H[6, 11]], [H[7, 3], H[7, 4], H[7, 5], H[7, 6], H[7, 7], H[7, 8], H[7, 9], H[7, 10], H[7, 11]], [H[8, 3], H[8, 4], H[8, 5], H[8, 6], H[8, 7], H[8, 8], H[8, 9], H[8, 10], H[8, 11]], [H[9, 3], H[9, 4], H[9, 5], H[9, 6], H[9, 7], H[9, 8], H[9, 9], H[9, 10], H[9, 11]], [H[10, 3], H[10, 4], H[10, 5], H[10, 6], H[10, 7], H[10, 8], H[10, 9], H[10, 10], H[10, 11]], [H[11, 3], H[11, 4], H[11, 5], H[11, 6], H[11, 7], H[11, 8], H[11, 9], H[11, 10], H[11, 11]]])
    return project_pd_9(eH) + ti.Matrix.identity(real, 9) * Q * Q
@ti.func
def PEM_energy(pos, posTilde, Q, r):
    a0 = ti.Vector([0.0, 0.0, 0.0])
    a1 = ti.Vector([pos[0], pos[1], pos[2]])
    b0 = ti.Vector([pos[3], pos[4], pos[5]])
    b1 = ti.Vector([pos[6], pos[7], pos[8]])
    _a0, _a1, _b0, _b1 = x0[PEM[r, 0]], x0[PEM[r, 1]], x0[PEM[r, 2]], x0[PEM[r, 3]]
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PE_3D_E(a0, b0, b1)
    return barrier_E(dist2, dHat2, kappa) * M_E(a0, a1, b0, b1, eps_x) + (pos - posTilde).norm_sqr() * Q * Q / 2
@ti.func
def PEM_gradient(pos, posTilde, Q, r):
    a0 = ti.Vector([0.0, 0.0, 0.0])
    a1 = ti.Vector([pos[0], pos[1], pos[2]])
    b0 = ti.Vector([pos[3], pos[4], pos[5]])
    b1 = ti.Vector([pos[6], pos[7], pos[8]])
    _a0, _a1, _b0, _b1 = x0[PEM[r, 0]], x0[PEM[r, 1]], x0[PEM[r, 2]], x0[PEM[r, 3]]
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PE_3D_E(a0, b0, b1)
    dist2g = PE_3D_g(a0, b0, b1)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    idx = ti.static([0, 1, 2, 6, 7, 8, 9, 10, 11])
    lg = fill_vec(bg * dist2g, idx, 9)
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    g = lg * M + b * Mg
    return ti.Vector([g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11]]) + (pos - posTilde) * Q * Q
@ti.func
def PEM_hessian(pos, posTilde, Q, r):
    a0 = ti.Vector([0.0, 0.0, 0.0])
    a1 = ti.Vector([pos[0], pos[1], pos[2]])
    b0 = ti.Vector([pos[3], pos[4], pos[5]])
    b1 = ti.Vector([pos[6], pos[7], pos[8]])
    _a0, _a1, _b0, _b1 = x0[PEM[r, 0]], x0[PEM[r, 1]], x0[PEM[r, 2]], x0[PEM[r, 3]]
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PE_3D_E(a0, b0, b1)
    dist2g = PE_3D_g(a0, b0, b1)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    idx = ti.static([0, 1, 2, 6, 7, 8, 9, 10, 11])
    lg = fill_vec(bg * dist2g, idx, 9)
    lH = fill_mat(barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_3D_H(a0, b0, b1), idx, 9)
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    H = lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
    eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5], H[3, 6], H[3, 7], H[3, 8], H[3, 9], H[3, 10], H[3, 11]], [H[4, 3], H[4, 4], H[4, 5], H[4, 6], H[4, 7], H[4, 8], H[4, 9], H[4, 10], H[4, 11]], [H[5, 3], H[5, 4], H[5, 5], H[5, 6], H[5, 7], H[5, 8], H[5, 9], H[5, 10], H[5, 11]], [H[6, 3], H[6, 4], H[6, 5], H[6, 6], H[6, 7], H[6, 8], H[6, 9], H[6, 10], H[6, 11]], [H[7, 3], H[7, 4], H[7, 5], H[7, 6], H[7, 7], H[7, 8], H[7, 9], H[7, 10], H[7, 11]], [H[8, 3], H[8, 4], H[8, 5], H[8, 6], H[8, 7], H[8, 8], H[8, 9], H[8, 10], H[8, 11]], [H[9, 3], H[9, 4], H[9, 5], H[9, 6], H[9, 7], H[9, 8], H[9, 9], H[9, 10], H[9, 11]], [H[10, 3], H[10, 4], H[10, 5], H[10, 6], H[10, 7], H[10, 8], H[10, 9], H[10, 10], H[10, 11]], [H[11, 3], H[11, 4], H[11, 5], H[11, 6], H[11, 7], H[11, 8], H[11, 9], H[11, 10], H[11, 11]]])
    return project_pd_9(eH) + ti.Matrix.identity(real, 9) * Q * Q


@ti.kernel
def local_elasticity():
    for e in range(n_elements):
        currentT = compute_T(e)
        Dx_plus_u_mtr = currentT @ restT[e].inverse() + u[e]
        U, sig, V = ti.svd(Dx_plus_u_mtr, real)
        sigma = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
        sigma_Dx_plus_u = sigma
        vol0 = restT[e].determinant() / dim / (dim - 1)
        for iter in range(20):
            g = local_gradient(sigma, sigma_Dx_plus_u, vol0, W[e])
            P = local_hessian(sigma, sigma_Dx_plus_u, vol0, W[e])
            p = -P.inverse() @ g
            alpha = 1.0
            sigma0 = sigma
            E0 = local_energy(sigma0, sigma_Dx_plus_u, vol0, W[e])
            sigma = sigma0 + p
            E = local_energy(sigma, sigma_Dx_plus_u, vol0, W[e])
            while E > E0:
                alpha *= 0.5
                sigma = sigma0 + alpha * p
                E = local_energy(sigma, sigma_Dx_plus_u, vol0, W[e])
        sig = ti.Matrix([[sigma[0], 0.0, 0.0], [0.0, sigma[1], 0.0], [0.0, 0.0, sigma[2]]])
        z[e] = U @ sig @ V.transpose()
@ti.kernel
def local_PP():
    for c in range(n_PP[None]):
        pos = ti.Vector([x(0)[PP[c, 0]] - x(0)[PP[c, 1]] + r_PP(0)[c, 0], x(1)[PP[c, 0]] - x(1)[PP[c, 1]] + r_PP(1)[c, 0], x(2)[PP[c, 0]] - x(2)[PP[c, 1]] + r_PP(2)[c, 0]])
        posTilde = pos
        for iter in range(20):
            g = PP_gradient(pos, posTilde, Q)
            P = PP_hessian(pos, posTilde, Q)
            p = -P.inverse() @ g
            alpha = 1.0
            pos0 = pos
            E0 = PP_energy(pos0, posTilde, Q)
            pos = pos0 + alpha * p
            E = PP_energy(pos, posTilde, Q)
            if iter == 19 and p.norm_sqr() > 1e-6:
                print("FATAL ERROR: Newton not converge")
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = PP_energy(pos, posTilde, Q)
        y_PP[c, 0] = ti.Vector([pos[0], pos[1], pos[2]])
@ti.kernel
def local_PE():
    for c in range(n_PE[None]):
        pos = ti.Vector([x(0)[PE[c, 0]] - x(0)[PE[c, 1]] + r_PE(0)[c, 0], x(1)[PE[c, 0]] - x(1)[PE[c, 1]] + r_PE(1)[c, 0], x(2)[PE[c, 0]] - x(2)[PE[c, 1]] + r_PE(2)[c, 0],
                         x(0)[PE[c, 0]] - x(0)[PE[c, 2]] + r_PE(0)[c, 1], x(1)[PE[c, 0]] - x(1)[PE[c, 2]] + r_PE(1)[c, 1], x(2)[PE[c, 0]] - x(2)[PE[c, 2]] + r_PE(2)[c, 1]])
        posTilde = pos
        for iter in range(20):
            g = PE_gradient(pos, posTilde, Q)
            P = PE_hessian(pos, posTilde, Q)
            p = -inverse_6(P) @ g
            alpha = 1.0
            pos0 = pos
            E0 = PE_energy(pos0, posTilde, Q)
            pos = pos0 + alpha * p
            E = PE_energy(pos, posTilde, Q)
            if iter == 19 and p.norm_sqr() > 1e-6:
                print("FATAL ERROR: Newton not converge")
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = PE_energy(pos, posTilde, Q)
        y_PE[c, 0], y_PE[c, 1] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]])
@ti.kernel
def local_PT():
    for c in range(n_PT[None]):
        pos = ti.Vector([x(0)[PT[c, 0]] - x(0)[PT[c, 1]] + r_PT(0)[c, 0], x(1)[PT[c, 0]] - x(1)[PT[c, 1]] + r_PT(1)[c, 0], x(2)[PT[c, 0]] - x(2)[PT[c, 1]] + r_PT(2)[c, 0],
                         x(0)[PT[c, 0]] - x(0)[PT[c, 2]] + r_PT(0)[c, 1], x(1)[PT[c, 0]] - x(1)[PT[c, 2]] + r_PT(1)[c, 1], x(2)[PT[c, 0]] - x(2)[PT[c, 2]] + r_PT(2)[c, 1],
                         x(0)[PT[c, 0]] - x(0)[PT[c, 3]] + r_PT(0)[c, 2], x(1)[PT[c, 0]] - x(1)[PT[c, 3]] + r_PT(1)[c, 2], x(2)[PT[c, 0]] - x(2)[PT[c, 3]] + r_PT(2)[c, 2]])
        posTilde = pos
        for iter in range(20):
            g = PT_gradient(pos, posTilde, Q)
            P = PT_hessian(pos, posTilde, Q)
            p = -inverse_9(P) @ g
            alpha = 1.0
            pos0 = pos
            E0 = PT_energy(pos0, posTilde, Q)
            pos = pos0 + alpha * p
            E = PT_energy(pos, posTilde, Q)
            if iter == 19 and p.norm_sqr() > 1e-6:
                print("FATAL ERROR: Newton not converge")
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = PT_energy(pos, posTilde, Q)
        y_PT[c, 0], y_PT[c, 1], y_PT[c, 2] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
@ti.kernel
def local_EE():
    for c in range(n_EE[None]):
        pos = ti.Vector([x(0)[EE[c, 0]] - x(0)[EE[c, 1]] + r_EE(0)[c, 0], x(1)[EE[c, 0]] - x(1)[EE[c, 1]] + r_EE(1)[c, 0], x(2)[EE[c, 0]] - x(2)[EE[c, 1]] + r_EE(2)[c, 0],
                         x(0)[EE[c, 0]] - x(0)[EE[c, 2]] + r_EE(0)[c, 1], x(1)[EE[c, 0]] - x(1)[EE[c, 2]] + r_EE(1)[c, 1], x(2)[EE[c, 0]] - x(2)[EE[c, 2]] + r_EE(2)[c, 1],
                         x(0)[EE[c, 0]] - x(0)[EE[c, 3]] + r_EE(0)[c, 2], x(1)[EE[c, 0]] - x(1)[EE[c, 3]] + r_EE(1)[c, 2], x(2)[EE[c, 0]] - x(2)[EE[c, 3]] + r_EE(2)[c, 2]])
        posTilde = pos
        for iter in range(20):
            g = EE_gradient(pos, posTilde, Q)
            P = EE_hessian(pos, posTilde, Q)
            p = -inverse_9(P) @ g
            alpha = 1.0
            pos0 = pos
            E0 = EE_energy(pos0, posTilde, Q)
            pos = pos0 + alpha * p
            E = EE_energy(pos, posTilde, Q)
            if iter == 19 and p.norm_sqr() > 1e-6:
                print("FATAL ERROR: Newton not converge")
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = EE_energy(pos, posTilde, Q)
        y_EE[c, 0], y_EE[c, 1], y_EE[c, 2] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
@ti.kernel
def local_EEM():
    for c in range(n_EEM[None]):
        pos = ti.Vector([x(0)[EEM[c, 0]] - x(0)[EEM[c, 1]] + r_EEM(0)[c, 0], x(1)[EEM[c, 0]] - x(1)[EEM[c, 1]] + r_EEM(1)[c, 0], x(2)[EEM[c, 0]] - x(2)[EEM[c, 1]] + r_EEM(2)[c, 0],
                         x(0)[EEM[c, 0]] - x(0)[EEM[c, 2]] + r_EEM(0)[c, 1], x(1)[EEM[c, 0]] - x(1)[EEM[c, 2]] + r_EEM(1)[c, 1], x(2)[EEM[c, 0]] - x(2)[EEM[c, 2]] + r_EEM(2)[c, 1],
                         x(0)[EEM[c, 0]] - x(0)[EEM[c, 3]] + r_EEM(0)[c, 2], x(1)[EEM[c, 0]] - x(1)[EEM[c, 3]] + r_EEM(1)[c, 2], x(2)[EEM[c, 0]] - x(2)[EEM[c, 3]] + r_EEM(2)[c, 2]])
        posTilde = pos
        for iter in range(20):
            g = EEM_gradient(pos, posTilde, Q, c)
            P = EEM_hessian(pos, posTilde, Q, c)
            p = -inverse_9(P) @ g
            alpha = 1.0
            pos0 = pos
            E0 = EEM_energy(pos0, posTilde, Q, c)
            pos = pos0 + alpha * p
            E = EEM_energy(pos, posTilde, Q, c)
            if iter == 19 and p.norm_sqr() > 1e-6:
                print("FATAL ERROR: Newton not converge")
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = EEM_energy(pos, posTilde, Q, c)
        y_EEM[c, 0], y_EEM[c, 1], y_EEM[c, 2] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
@ti.kernel
def local_PPM():
    for c in range(n_PPM[None]):
        pos = ti.Vector([x(0)[PPM[c, 0]] - x(0)[PPM[c, 1]] + r_PPM(0)[c, 0], x(1)[PPM[c, 0]] - x(1)[PPM[c, 1]] + r_PPM(1)[c, 0], x(2)[PPM[c, 0]] - x(2)[PPM[c, 1]] + r_PPM(2)[c, 0],
                         x(0)[PPM[c, 0]] - x(0)[PPM[c, 2]] + r_PPM(0)[c, 1], x(1)[PPM[c, 0]] - x(1)[PPM[c, 2]] + r_PPM(1)[c, 1], x(2)[PPM[c, 0]] - x(2)[PPM[c, 2]] + r_PPM(2)[c, 1],
                         x(0)[PPM[c, 0]] - x(0)[PPM[c, 3]] + r_PPM(0)[c, 2], x(1)[PPM[c, 0]] - x(1)[PPM[c, 3]] + r_PPM(1)[c, 2], x(2)[PPM[c, 0]] - x(2)[PPM[c, 3]] + r_PPM(2)[c, 2]])
        posTilde = pos
        for iter in range(20):
            g = PPM_gradient(pos, posTilde, Q, c)
            P = PPM_hessian(pos, posTilde, Q, c)
            p = -inverse_9(P) @ g
            alpha = 1.0
            pos0 = pos
            E0 = PPM_energy(pos0, posTilde, Q, c)
            pos = pos0 + alpha * p
            E = PPM_energy(pos, posTilde, Q, c)
            if iter == 19 and p.norm_sqr() > 1e-6:
                print("FATAL ERROR: Newton not converge")
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = PPM_energy(pos, posTilde, Q, c)
        y_PPM[c, 0], y_PPM[c, 1], y_PPM[c, 2] = ti.Vector([pos[0], pos[1], pos[2]]), ti.Vector([pos[3], pos[4], pos[5]]), ti.Vector([pos[6], pos[7], pos[8]])
@ti.kernel
def local_PEM():
    for c in range(n_PEM[None]):
        pos = ti.Vector([x(0)[PEM[c, 0]] - x(0)[PEM[c, 1]] + r_PEM(0)[c, 0], x(1)[PEM[c, 0]] - x(1)[PEM[c, 1]] + r_PEM(1)[c, 0], x(2)[PEM[c, 0]] - x(2)[PEM[c, 1]] + r_PEM(2)[c, 0],
                         x(0)[PEM[c, 0]] - x(0)[PEM[c, 2]] + r_PEM(0)[c, 1], x(1)[PEM[c, 0]] - x(1)[PEM[c, 2]] + r_PEM(1)[c, 1], x(2)[PEM[c, 0]] - x(2)[PEM[c, 2]] + r_PEM(2)[c, 1],
                         x(0)[PEM[c, 0]] - x(0)[PEM[c, 3]] + r_PEM(0)[c, 2], x(1)[PEM[c, 0]] - x(1)[PEM[c, 3]] + r_PEM(1)[c, 2], x(2)[PEM[c, 0]] - x(2)[PEM[c, 3]] + r_PEM(2)[c, 2]])
        posTilde = pos
        for iter in range(20):
            g = PEM_gradient(pos, posTilde, Q, c)
            P = PEM_hessian(pos, posTilde, Q, c)
            p = -inverse_9(P) @ g
            alpha = 1.0
            pos0 = pos
            E0 = PEM_energy(pos0, posTilde, Q, c)
            pos = pos0 + alpha * p
            E = PEM_energy(pos, posTilde, Q, c)
            if iter == 19 and p.norm_sqr() > 1e-6:
                print("FATAL ERROR: Newton not converge")
            while E > E0:
                alpha *= 0.5
                pos = pos0 + alpha * p
                E = PEM_energy(pos, posTilde, Q, c)
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
        currentT = compute_T(i)
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
def compute_v():
    for i in range(n_particles):
        v[i] = (x[i] - xn[i]) / dt


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
        model.vi.from_numpy(particle_pos.astype(np.float32))
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
    x.from_numpy(mesh_particles.astype(np.float32))
    v.fill(0)
    vertices.from_numpy(mesh_elements.astype(np.int32))
    boundary_points.from_numpy(np.array(list(boundary_points_)).astype(np.int32))
    boundary_edges.from_numpy(boundary_edges_.astype(np.int32))
    boundary_triangles.from_numpy(boundary_triangles_.astype(np.int32))
    compute_restT_and_m()
    vertices_ = vertices.to_numpy()
    write_image(0)
    total_time = 0
    for f in range(180):
        total_time -= time.time()
        print("==================== Frame: ", f, " ====================")
        initial_guess()
        prs = []
        drs = []
        for step in range(20):
            find_constraints()

            data_row.fill(0)
            data_col.fill(0)
            data_val.fill(0)
            data_rhs.fill(0)
            data_x.fill(0)

            global_step()
            global_PP()
            global_PE()
            global_PT()
            global_EE()
            global_EEM()
            global_PPM()
            global_PEM()

            solve_system()

            local_elasticity()
            local_PP()
            local_PE()
            local_PT()
            local_EE()
            local_EEM()
            local_PPM()
            local_PEM()

            # pr = prime_residual()
            # prs.append(math.log(max(pr, 1e-20)))
            # dr = dual_residual()
            # drs.append(math.log(max(dr, 1e-20)))
            # xr = X_residual()
            # print(f, "/", step, f" change of X: {xr:.8f}, prime residual: {pr:.8f}, dual residual: {dr:.8f}")

            dual_step()
        # print(f, sha1(x.to_numpy()).hexdigest())

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
        total_time += time.time()
        print("Time : ", total_time)
        write_image(f + 1)
    cmd = 'ffmpeg -framerate 12 -i "' + directory + 'images/%6d.png" -c:v libx264 -profile:v high -crf 10 -pix_fmt yuv420p -threads 20 ' + directory + 'video.mp4'
    os.system((cmd))
