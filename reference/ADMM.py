import sys
sys.path.insert(0, "../../build")
from JGSL_WATER import *
import taichi as ti
import numpy as np
import pymesh
from fixed_corotated import *
from math_tools import *

##############################################################################

mesh = pymesh.load_mesh("input/Sharkey.obj")
mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.6
mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 0.9

##############################################################################

ti.init(arch=ti.cpu)

real = ti.f32
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
n_particles = mesh.num_vertices
n_elements = mesh.num_faces
cnt = ti.var(dt=ti.i32, shape=())

x, xTilde, xn, v, m = vec(), vec(), vec(), vec(), scalar()
restT = mat()
vertices = ti.var(ti.i32)
W, z, zz, u = scalar(), mat(), mat(), mat()
ti.root.dense(ti.k, n_particles).place(x, xTilde, xn, v, m)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.ij, (n_elements, 3)).place(vertices)
ti.root.dense(ti.i, n_elements).place(W, z, zz, u)

data_rhs = ti.var(real, shape=2000)
data_mat = ti.var(real, shape=(3, 100000))
data_x = ti.var(real, shape=2000)


@ti.func
def compute_T(i):
    a = vertices[i, 0]
    b = vertices[i, 1]
    c = vertices[i, 2]
    ab = x[b] - x[a]
    ac = x[c] - x[a]
    return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])


@ti.kernel
def compute_restT_and_m():
    for i in range(n_elements):
        restT[i] = compute_T(i)
        mass = restT[i].determinant() / 2 * density / 3
        if mass < 0.0:
            print("FATAL ERROR : mesh inverted")
        for d in ti.static(range(3)):
            m[vertices[i, d]] += mass


@ti.kernel
def initial_guess():
    # set W, u, z
    for i in range(n_elements):
        currentT = compute_T(i)
        W[i] = ti.sqrt(la + mu * 2 / 3) * (restT[i].determinant() / 2)
        z[i] = currentT @ restT[i].inverse()
        u[i] = ti.Matrix([[0, 0], [0, 0]])
    for i in range(n_particles):
        xn[i] = x[i]
        xTilde[i] = x[i] + dt * v[i]
        xTilde(1)[i] -= dt * dt * 9.8


@ti.func
def X2F(p, q, i, j, A):
    ret = 0.0
    if p == 0 and j == 0:
        ret = -A[0, 0] - A[1, 0]
    if p == 0 and j == 1:
        ret = -A[0, 1] - A[1, 1]
    if p == 1 and j == 0:
        ret = A[0, 0]
    if p == 1 and j == 1:
        ret = A[0, 1]
    if p == 2 and j == 0:
        ret = A[1, 0]
    if p == 2 and j == 1:
        ret = A[1, 1]
    return ret


@ti.kernel
def global_step():
    cnt[None] = 0
    for i in range(n_particles):
        for d in ti.static(range(2)):
            c = i * 2 + d
            data_mat[0, c] = i * 2 + d
            data_mat[1, c] = i * 2 + d
            data_mat[2, c] = m[i]
            data_rhs[i * 2 + d] += m[i] * xTilde(d)[i]
    cnt[None] += n_particles * 2
    for e in range(n_elements):
        A = restT[e].inverse()
        for p in ti.static(range(3)):
            for i in ti.static(range(2)):
                for j in ti.static(range(2)):
                    for pp in ti.static(range(3)):
                        q, qq = i, i
                        c = cnt[None] + e * 36 + p * 12 + i * 6 + j * 3 + pp
                        data_mat[0, c] = vertices[e, p] * 2 + q
                        data_mat[1, c] = vertices[e, pp] * 2 + qq
                        data_mat[2, c] = X2F(p, q, i, j, A) * X2F(pp, qq, i, j, A) * W[e] * W[e]
        F = z[e] - u[e]
        for p in ti.static(range(3)):
            for i in ti.static(range(2)):
                for j in ti.static(range(2)):
                    q = i
                    data_rhs[vertices[e, p] * 2 + q] += X2F(p, q, i, j, A) * F[i, j] * W[e] * W[e]
    cnt[None] += n_elements * 36


@ti.func
def local_energy(sigma, sigma_Dx_plus_u, vol0, W):
    return fixed_corotated_energy(sigma, la, mu) * dt * dt * vol0 + (sigma - sigma_Dx_plus_u).norm_sqr() * W * W / 2


@ti.func
def local_gradient(sigma, sigma_Dx_plus_u, vol0, W):
    return fixed_corotated_gradient(sigma, la, mu) * dt * dt * vol0 + (sigma - sigma_Dx_plus_u) * W * W


@ti.func
def local_hessian(sigma, sigma_Dx_plus_u, vol0, W):
    return make_pd(fixed_corotated_hessian(sigma, la, mu)) * dt * dt * vol0 + ti.Matrix([[1, 0], [0, 1]]) * W * W


@ti.kernel
def local_step():
    for i in range(n_particles):
        for d in ti.static(range(2)):
            x(d)[i] = data_x[i * 2 + d]
    for e in range(n_elements):
        currentT = compute_T(e)
        Dx_plus_u_mtr = currentT @ restT[e].inverse() + u[e]
        U, sig, V = ti.svd(Dx_plus_u_mtr, real)
        sigma = ti.Vector([sig[0, 0], sig[1, 1]])
        sigma_Dx_plus_u = sigma
        vol0 = restT[e].determinant() / 2
        for iter in ti.static(range(20)):
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
        z[e] = U @ ti.Matrix([[sigma[0], 0], [0, sigma[1]]]) @ V.transpose()


@ti.kernel
def output_residual():
    # reuse data_rhs
    residual1 = 0.0
    for i in range(n_elements):
        currentT = compute_T(i)
        F = currentT @ restT[i].inverse()
        residual1 += (F - z[i]).norm_sqr() * W[i] * W[i]
    residual2 = 0.0
    for e in range(n_elements):
        A = restT[e].inverse()
        delta = z[e] - zz[e]
        for p in ti.static(range(3)):
            for i in ti.static(range(2)):
                for j in ti.static(range(2)):
                    q = i
                    data_rhs[vertices[e, p] * 2 + q] += X2F(p, q, i, j, A) * delta[i, j] * W[e] * W[e]
        zz[e] = z[e]
    for i in range(n_particles * 2):
        residual2 += data_rhs[i] * data_rhs[i]
    print("Primal Residual : ", residual1, " Dual Residual : ", residual2)

@ti.kernel
def dual_step():
    for i in range(n_elements):
        currentT = compute_T(i)
        F = currentT @ restT[i].inverse()
        u[i] += F - z[i]


@ti.kernel
def compute_v():
    for i in range(n_particles):
        v[i] = (x[i] - xn[i]) / dt


if __name__ == "__main__":
    x.from_numpy(mesh.vertices.astype(np.float32))
    v.fill(0)
    vertices.from_numpy(mesh.faces)
    compute_restT_and_m()
    gui = ti.GUI("MPM", (1024, 1024), background_color=0x112F41)
    vertices_ = vertices.to_numpy()
    for f in range(360):
        print("==================== Frame: ", f, " ====================")
        initial_guess()
        for i in range(20):
            data_mat.fill(0)
            data_rhs.fill(0)
            data_x.fill(0)
            global_step()
            data_x.from_numpy(solve_linear_system(data_mat.to_numpy(), data_rhs.to_numpy(), n_particles * 2, np.array([i for i in range(12)]), xn.to_numpy(), False, 0, cnt[None]))
            local_step()
            data_rhs.fill(0)
            output_residual()
            dual_step()
        compute_v()
        # TODO: why is visualization so slow?
        particle_pos = (x.to_numpy() + mesh_offset) * mesh_scale
        for i in range(n_elements):
            for j in range(3):
                a, b = vertices_[i, j], vertices_[i, (j + 1) % 3]
                gui.line((particle_pos[a][0], particle_pos[a][1]),
                         (particle_pos[b][0], particle_pos[b][1]),
                         radius=1,
                         color=0x4FB99F)
        gui.show(f'output/{f:06d}.png')