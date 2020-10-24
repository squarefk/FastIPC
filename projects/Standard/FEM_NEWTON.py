import sys
import taichi as ti
import numpy as np
import pymesh
import scipy.sparse
import scipy.sparse.linalg
from common.physics.fixed_corotated import *
from common.math.math_tools import *
import meshio

##############################################################################

mesh = meshio.read("../FastIPC/input/Sharkey.obj")
mesh_particles = mesh.points
mesh_elements = mesh.cells[0].data
mesh_scale = 0.6
mesh_offset = [0.35, 0.3]

##############################################################################

real = ti.f64
ti.init(arch=ti.cpu, default_fp=real)

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

dim = 2
dt = 0.01
E = 1e4
nu = 0.4
la = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
density = 100
n_particles = len(mesh_particles)
n_elements = len(mesh_elements)
cnt = ti.var(dt=ti.i32, shape=())

x, xPrev, xTilde, xn, v, m = vec(), vec(), vec(), vec(), vec(), scalar()
zero = vec()
restT = mat()
vertices = ti.var(ti.i32)
ti.root.dense(ti.k, n_particles).place(x, xPrev, xTilde, xn, v, m)
ti.root.dense(ti.k, n_particles).place(zero)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.ij, (n_elements, dim + 1)).place(vertices)

data_rhs = ti.var(real, shape=n_particles * dim)
data_mat = ti.var(real, shape=(3, 2000000))
data_sol = ti.var(real, shape=n_particles * dim)


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
        xTilde(1)[i] -= dt * dt * 9.8


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
        U, sig, V = svd(F)
        total_energy += elasticity_energy(sig, la, mu) * dt * dt * vol0
    return total_energy


@ti.kernel
def compute_hessian_and_gradient():
    cnt[None] = 0
    # inertia
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            c = cnt[None] + i * dim + d
            data_mat[0, c] = i * dim + d
            data_mat[1, c] = i * dim + d
            data_mat[2, c] = m[i]
            data_rhs[i * dim + d] -= m[i] * (x(d)[i] - xTilde(d)[i])
    cnt[None] += n_particles * dim
    # elasticity
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
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[2], indMap[colI], _000 + _101
                c = cnt[None] + e * 36 + colI * 6 + 1
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[3], indMap[colI], _200 + _301
                c = cnt[None] + e * 36 + colI * 6 + 2
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[4], indMap[colI], _010 + _111
                c = cnt[None] + e * 36 + colI * 6 + 3
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[5], indMap[colI], _210 + _311
                c = cnt[None] + e * 36 + colI * 6 + 4
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[0], indMap[colI], - _000 - _101 - _010 - _111
                c = cnt[None] + e * 36 + colI * 6 + 5
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[1], indMap[colI], - _200 - _301 - _210 - _311
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
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[3], IB[0, 0] * intermediate[rowI, 0] + IB[0, 1] * intermediate[rowI, 3] + IB[0, 2] * intermediate[rowI, 6]
                c = cnt[None] + e * 144 + rowI * 12 + 1
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[4], IB[0, 0] * intermediate[rowI, 1] + IB[0, 1] * intermediate[rowI, 4] + IB[0, 2] * intermediate[rowI, 7]
                c = cnt[None] + e * 144 + rowI * 12 + 2
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[5], IB[0, 0] * intermediate[rowI, 2] + IB[0, 1] * intermediate[rowI, 5] + IB[0, 2] * intermediate[rowI, 8]
                c = cnt[None] + e * 144 + rowI * 12 + 3
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[6], IB[1, 0] * intermediate[rowI, 0] + IB[1, 1] * intermediate[rowI, 3] + IB[1, 2] * intermediate[rowI, 6]
                c = cnt[None] + e * 144 + rowI * 12 + 4
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[7], IB[1, 0] * intermediate[rowI, 1] + IB[1, 1] * intermediate[rowI, 4] + IB[1, 2] * intermediate[rowI, 7]
                c = cnt[None] + e * 144 + rowI * 12 + 5
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[8], IB[1, 0] * intermediate[rowI, 2] + IB[1, 1] * intermediate[rowI, 5] + IB[1, 2] * intermediate[rowI, 8]
                c = cnt[None] + e * 144 + rowI * 12 + 6
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[9], IB[2, 0] * intermediate[rowI, 0] + IB[2, 1] * intermediate[rowI, 3] + IB[2, 2] * intermediate[rowI, 6]
                c = cnt[None] + e * 144 + rowI * 12 + 7
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[10], IB[2, 0] * intermediate[rowI, 1] + IB[2, 1] * intermediate[rowI, 4] + IB[2, 2] * intermediate[rowI, 7]
                c = cnt[None] + e * 144 + rowI * 12 + 8
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[11], IB[2, 0] * intermediate[rowI, 2] + IB[2, 1] * intermediate[rowI, 5] + IB[2, 2] * intermediate[rowI, 8]
                c = cnt[None] + e * 144 + rowI * 12 + 9
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[0], -data_mat[2, c - 9] - data_mat[2, c - 6] - data_mat[2, c - 3]
                c = cnt[None] + e * 144 + rowI * 12 + 10
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[1], -data_mat[2, c - 9] - data_mat[2, c - 6] - data_mat[2, c - 3]
                c = cnt[None] + e * 144 + rowI * 12 + 11
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[2], -data_mat[2, c - 9] - data_mat[2, c - 6] - data_mat[2, c - 3]
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


if __name__ == "__main__":
    x.from_numpy(mesh_particles)
    v.fill(0)
    vertices.from_numpy(mesh_elements)
    compute_restT_and_m()
    if dim == 2:
        gui = ti.GUI("MPM", (1024, 1024), background_color=0x112F41)
    zero.fill(0)
    for f in range(360):
        print("==================== Frame: ", f, " ====================")
        compute_xn_and_xTilde()
        while True:
            data_mat.fill(0)
            data_rhs.fill(0)
            data_sol.fill(0)
            compute_hessian_and_gradient()

            print("Total entries: ", cnt[None])
            mat = data_mat.to_numpy()
            row, col, val = mat[0, :cnt[None]], mat[1, :cnt[None]], mat[2, :cnt[None]]
            rhs = data_rhs.to_numpy()
            n = n_particles * dim
            A = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n))
            A = scipy.sparse.lil_matrix(A)
            D = np.array([i for i in range(12 * dim)])
            A[:, D] = 0
            A[D, :] = 0
            A = scipy.sparse.csr_matrix(A)
            A += scipy.sparse.csr_matrix((np.ones(len(D)), (D, D)), shape=(n, n))
            rhs[D] = 0
            data_sol.from_numpy(scipy.sparse.linalg.spsolve(A, rhs))

            print('residual : ', output_residual())
            if output_residual() < 1e-2 * dt:
                break
            E0 = compute_energy()
            save_xPrev()
            alpha = 1.0
            apply_sol(alpha)
            E = compute_energy()
            while E > E0:
                alpha *= 0.5
                apply_sol(alpha)
                E = compute_energy()
        compute_v()
        # TODO: why is visualization so slow?
        particle_pos = x.to_numpy() * mesh_scale + mesh_offset
        vertices_ = vertices.to_numpy()
        if dim == 2:
            for i in range(n_elements):
                for j in range(3):
                    a, b = vertices_[i, j], vertices_[i, (j + 1) % 3]
                    gui.line((particle_pos[a][0], particle_pos[a][1]),
                             (particle_pos[b][0], particle_pos[b][1]),
                             radius=1,
                             color=0x4FB99F)
            gui.show(f'output/{f:06d}.png')
        else:
            f = open(f'output/{f:06d}.obj', 'w')
            for i in range(n_particles):
                f.write('v %.6f %.6f %.6f\n' % (particle_pos[i, 0], particle_pos[i, 1], particle_pos[i, 2]))
            for [p0, p1, p2] in boundary_triangles_:
                f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
            f.close()
