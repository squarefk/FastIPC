from hashlib import sha1
import sys, os, time, math
sys.path.insert(0, "../../build")
from JGSL_WATER import *
import taichi as ti
import numpy as np
import pymesh
import matplotlib.pyplot as plt
from fixed_corotated import *
from math_tools import *
from ipc import *
from reader import *

##############################################################################

mesh, dirichlet, mesh_scale, mesh_offset = read(int(sys.argv[1]))
Q_weight = float(sys.argv[2])
edges = set()
for [i, j, k] in mesh.faces:
    edges.add((i, j))
    edges.add((j, k))
    edges.add((k, i))
boundary_points_ = set()
boundary_edges_ = np.zeros(shape=(0, 2), dtype=np.int32)
for [i, j, k] in mesh.faces:
    if (j, i) not in edges:
        boundary_points_.update([j, i])
        boundary_edges_ = np.vstack((boundary_edges_, [j, i]))
    if (k, j) not in edges:
        boundary_points_.update([k, j])
        boundary_edges_ = np.vstack((boundary_edges_, [k, j]))
    if (i, k) not in edges:
        boundary_points_.update([i, k])
        boundary_edges_ = np.vstack((boundary_edges_, [i, k]))
boundary_edges_ = np.vstack((boundary_edges_, [0, 1]))

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

dim = 2
dt = 0.01
E = 1e4
nu = 0.4
la = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
density = 100
n_particles = mesh.num_vertices
n_elements = mesh.num_faces
n_boundary_points = len(boundary_points_)
n_boundary_edges = len(boundary_edges_)

x, xx, xTilde, xn, v, m = vec(), vec(), vec(), vec(), vec(), scalar()
restT = mat()
vertices = ti.var(ti.i32)
W, z, zz, u = scalar(), mat(), mat(), mat()
boundary_points = ti.var(ti.i32)
boundary_edges = ti.var(ti.i32)
ti.root.dense(ti.k, n_particles).place(x, xx, xTilde, xn, v, m)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.ij, (n_elements, 3)).place(vertices)
ti.root.dense(ti.i, n_elements).place(W, z, zz, u)
ti.root.dense(ti.i, n_boundary_points).place(boundary_points)
ti.root.dense(ti.ij, (n_boundary_edges, 2)).place(boundary_edges)

data_rhs = ti.var(real, shape=2000)
data_mat = ti.var(real, shape=(3, 1000000))
data_x = ti.var(real, shape=2000)
cnt = ti.var(dt=ti.i32, shape=())

n_constraint = 1000000
constraints = ti.var(ti.i32, shape=(n_constraint, 3))
old_constraints = ti.var(ti.i32, shape=(n_constraint, 3))
cc = ti.var(dt=ti.i32, shape=())
old_cc = ti.var(dt=ti.i32, shape=())
y, yy, r, old_y, old_r = vec(), vec(), vec(), vec(), vec()
ti.root.dense(ti.ij, (n_constraint, 2)).place(y, yy, r, old_y, old_r)
Q, old_Q = scalar(), scalar()
ti.root.dense(ti.i, n_constraint).place(Q, old_Q)

dHat2 = 1e-5
dHat = dHat2 ** 0.5
kappa = 1e4


@ti.kernel
def find_constraints():
    old_cc[None] = cc[None]
    cc[None] = 0
    for c in range(old_cc[None]):
        old_constraints[c, 0] = constraints[c, 0]
        old_constraints[c, 1] = constraints[c, 1]
        old_constraints[c, 2] = constraints[c, 2]
        old_y[c, 0], old_y[c, 1] = y[c, 0], y[c, 1]
        old_r[c, 0], old_r[c, 1] = r[c, 0], r[c, 1]
        old_Q[c] = Q[c]
    cc[None] = 0
    for _ in range(1):
        for i in range(len(boundary_points_)):
            p0 = boundary_points[i]
            for j in range(n_boundary_edges):
                e0 = boundary_edges[j, 0]
                e1 = boundary_edges[j, 1]
                if p0 != e0 and p0 != e1:
                    if point_edge_ccd_broadphase(x[p0], x[e0], x[e1], dHat):
                        if ipc_energy(x[p0], x[e0], x[e1], dHat2, kappa) > 0:
                            c = ti.atomic_add(cc[None], 1)
                            constraints[c, 0] = p0
                            constraints[c, 1] = e0
                            constraints[c, 2] = e1
                            Q[c] = Q_weight
                            y[c, 0], y[c, 1] = x[p0] - x[e0], x[p0] - x[e1]
                            r[c, 0], r[c, 1] = ti.Vector([0.0, 0.0]), ti.Vector([0.0, 0.0])
    for c in range(old_cc[None]):
        for d in range(cc[None]):
            if old_constraints[c, 0] == constraints[d, 0] and old_constraints[c, 1] == constraints[d, 1] and old_constraints[c, 2] == constraints[d, 2]:
                y[d ,0], y[d, 1] = old_y[c, 0], old_y[c, 1]
                r[d ,0], r[d, 1] = old_r[c, 0], old_r[c, 1]


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
    for _ in range(1):
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
    for c in range(cc[None]):
        p0 = constraints[c, 0]
        e0 = constraints[c, 1]
        e1 = constraints[c, 2]
        y[c, 0], y[c, 1] = xTilde[p0] - xTilde[e0], xTilde[p0] - xTilde[e1]
        r[c, 0], r[c, 1] = ti.Vector([0.0, 0.0]), ti.Vector([0.0, 0.0])


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
    for _ in range(1):
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
    ETE = ti.Matrix([[2, -1, -1], [-1, 1, 0], [-1, 0, 1]])
    for _ in range(1):
        for c in range(cc[None]):
            for p in ti.static(range(3)):
                for q in ti.static(range(3)):
                    for j in ti.static(range(2)):
                        idx = cnt[None] + c * 18 + p * 6 + q * 2 + j
                        data_mat[0, idx] = constraints[c, p] * 2 + j
                        data_mat[1, idx] = constraints[c, q] * 2 + j
                        data_mat[2, idx] = ETE[p, q] * Q[c] * Q[c]
            for j in ti.static(range(2)):
                data_rhs[constraints[c, 0] * 2 + j] += (y(j)[c, 0] - r(j)[c, 0]) * Q[c] * Q[c]
                data_rhs[constraints[c, 0] * 2 + j] += (y(j)[c, 1] - r(j)[c, 1]) * Q[c] * Q[c]
                data_rhs[constraints[c, 1] * 2 + j] -= (y(j)[c, 0] - r(j)[c, 0]) * Q[c] * Q[c]
                data_rhs[constraints[c, 2] * 2 + j] -= (y(j)[c, 1] - r(j)[c, 1]) * Q[c] * Q[c]
    cnt[None] += cc[None] * 18


@ti.kernel
def apply_newton_result():
    for i in range(n_particles):
        for d in ti.static(range(2)):
            x(d)[i] = data_x[i * 2 + d]


@ti.func
def local_energy(sigma, sigma_Dx_plus_u, vol0, W):
    return fixed_corotated_energy(sigma, la, mu) * dt * dt * vol0 + (sigma - sigma_Dx_plus_u).norm_sqr() * W * W / 2


@ti.func
def local_gradient(sigma, sigma_Dx_plus_u, vol0, W):
    return fixed_corotated_gradient(sigma, la, mu) * dt * dt * vol0 + (sigma - sigma_Dx_plus_u) * W * W


@ti.func
def local_hessian(sigma, sigma_Dx_plus_u, vol0, W):
    return make_pd(fixed_corotated_hessian(sigma, la, mu)) * dt * dt * vol0 + ti.Matrix([[1, 0], [0, 1]]) * W * W


@ti.func
def second_energy(pos, posTilde, Q):
    return ipc_energy(ti.Vector([0.0, 0.0]), ti.Vector([pos[0], pos[1]]), ti.Vector([pos[2], pos[3]]), dHat2, kappa) + (pos - posTilde).norm_sqr() * Q * Q / 2


@ti.func
def second_gradient(pos, posTilde, Q):
    g = ipc_gradient(ti.Vector([0.0, 0.0]), ti.Vector([pos[0], pos[1]]), ti.Vector([pos[2], pos[3]]), dHat2, kappa)
    return ti.Vector([g[2], g[3], g[4], g[5]]) + (pos - posTilde) * Q * Q


@ti.func
def second_hessian(pos, posTilde, Q):
    return project_pd64(ipc_hessian(ti.Vector([0.0, 0.0]), ti.Vector([pos[0], pos[1]]), ti.Vector([pos[2], pos[3]]), dHat2, kappa), Q * Q)


@ti.kernel
def local_step():
    for e in range(n_elements):
        currentT = compute_T(e)
        Dx_plus_u_mtr = currentT @ restT[e].inverse() + u[e]
        U, sig, V = ti.svd(Dx_plus_u_mtr, real)
        sigma = ti.Vector([sig[0, 0], sig[1, 1]])
        sigma_Dx_plus_u = sigma
        vol0 = restT[e].determinant() / 2
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
        z[e] = U @ ti.Matrix([[sigma[0], 0], [0, sigma[1]]]) @ V.transpose()
    for c in range(cc[None]):
        pos = ti.Vector([x(0)[constraints[c, 0]] - x(0)[constraints[c, 1]] + r(0)[c, 0], x(1)[constraints[c, 0]] - x(1)[constraints[c, 1]] + r(1)[c, 0],
                         x(0)[constraints[c, 0]] - x(0)[constraints[c, 2]] + r(0)[c, 1], x(1)[constraints[c, 0]] - x(1)[constraints[c, 2]] + r(1)[c, 1]])
        posTilde = pos
        if not ipc_overlap(ti.Vector([0.0, 0.0]), ti.Vector([pos[0], pos[1]]), ti.Vector([pos[2], pos[3]])):
            for iter in range(20):
                g = second_gradient(pos, posTilde, Q[c])
                P = second_hessian(pos, posTilde, Q[c])
                p = -P @ g
                alpha = 1.0

                x0, x1, x2 = ti.Vector([0.0, 0.0]), ti.Vector([pos[0], pos[1]]), ti.Vector([pos[2], pos[3]])
                d0, d1, d2 = ti.Vector([0.0, 0.0]), ti.Vector([p[0], p[1]]), ti.Vector([p[2], p[3]])
                if moving_point_edge_ccd_broadphase(x0, x1, x2, d0, d1, d2, dHat):
                    alpha = ti.min(alpha, moving_point_edge_ccd(x0, x1, x2, d0, d1, d2, 0.1))

                pos0 = pos
                E0 = second_energy(pos0, posTilde, Q[c])
                pos = pos0 + alpha * p
                E = second_energy(pos, posTilde, Q[c])
                if iter == 19 and p.norm_sqr() > 1e-6:
                    print("FATAL ERROR: Newton not converge")
                while E > E0:
                    alpha *= 0.5
                    pos = pos0 + alpha * p
                    E = second_energy(pos, posTilde, Q[c])
        y[c, 0], y[c, 1] = ti.Vector([pos[0], pos[1]]), ti.Vector([pos[2], pos[3]])


@ti.kernel
def prime_residual() -> real:
    residual = 0.0
    for i in range(n_elements):
        currentT = compute_T(i)
        F = currentT @ restT[i].inverse()
        residual += (F - z[i]).norm_sqr() * W[i] * W[i]
    for c in range(cc[None]):
        residual += (x[constraints[c, 0]] - x[constraints[c, 1]] - y[c, 0]).norm_sqr() * Q[c] * Q[c]
        residual += (x[constraints[c, 0]] - x[constraints[c, 2]] - y[c, 1]).norm_sqr() * Q[c] * Q[c]
    return residual


@ti.kernel
def dual_residual() -> real:
    residual = 0.0
    for i in data_rhs:
        data_rhs[i] = 0
    for e in range(n_elements):
        A = restT[e].inverse()
        delta = z[e] - zz[e]
        for p in ti.static(range(3)):
            for i in ti.static(range(2)):
                for j in ti.static(range(2)):
                    q = i
                    data_rhs[vertices[e, p] * 2 + q] += X2F(p, q, i, j, A) * delta[i, j] * W[e] * W[e]
        zz[e] = z[e]
    for i in data_rhs:
        residual += data_rhs[i] * data_rhs[i]

    for i in data_rhs:
        data_rhs[i] = 0

    for c in range(old_cc[None]):
        for j in ti.static(range(2)):
            data_rhs[constraints[c, 0] * 2 + j] += (- old_y(j)[c, 0]) * old_Q[c] * old_Q[c]
            data_rhs[constraints[c, 0] * 2 + j] += (- old_y(j)[c, 1]) * old_Q[c] * old_Q[c]
            data_rhs[constraints[c, 1] * 2 + j] -= (- old_y(j)[c, 0]) * old_Q[c] * old_Q[c]
            data_rhs[constraints[c, 2] * 2 + j] -= (- old_y(j)[c, 1]) * old_Q[c] * old_Q[c]
    for d in range(cc[None]):
        for j in ti.static(range(2)):
            data_rhs[constraints[d, 0] * 2 + j] += (y(j)[d, 0]) * Q[d] * Q[d]
            data_rhs[constraints[d, 0] * 2 + j] += (y(j)[d, 1]) * Q[d] * Q[d]
            data_rhs[constraints[d, 1] * 2 + j] -= (y(j)[d, 0]) * Q[d] * Q[d]
            data_rhs[constraints[d, 2] * 2 + j] -= (y(j)[d, 1]) * Q[d] * Q[d]
    for i in data_rhs:
        residual += data_rhs[i] * data_rhs[i]
    return residual


@ti.kernel
def X_residual() -> real:
    residual = 0.0
    for _ in range(1):
        for i in range(n_particles):
            residual = max(residual, (xx[i] - x[i]).norm_sqr())
            xx[i] = x[i]
    return residual


@ti.kernel
def dual_step():
    for i in range(n_elements):
        currentT = compute_T(i)
        F = currentT @ restT[i].inverse()
        u[i] += F - z[i]
    for c in range(cc[None]):
        r[c, 0] += x[constraints[c, 0]] - x[constraints[c, 1]] - y[c, 0]
        r[c, 1] += x[constraints[c, 0]] - x[constraints[c, 2]] - y[c, 1]


@ti.kernel
def compute_v():
    for i in range(n_particles):
        v[i] = (x[i] - xn[i]) / dt


def write_image(f):
    find_constraints()
    particle_pos = (x.to_numpy() + mesh_offset) * mesh_scale
    gui.line((particle_pos[0][0], particle_pos[0][1]),
             (particle_pos[1][0], particle_pos[1][1]),
             radius=1,
             color=0x4FB99F)
    for i in range(n_elements):
        for j in range(3):
            a, b = vertices_[i, j], vertices_[i, (j + 1) % 3]
            gui.line((particle_pos[a][0], particle_pos[a][1]),
                     (particle_pos[b][0], particle_pos[b][1]),
                     radius=1,
                     color=0x4FB99F)
    for i in dirichlet:
        gui.circle(particle_pos[i], radius=3, color=0x44FFFF)
    for i in range(cc[None]):
        gui.circle(particle_pos[constraints[i, 0]], radius=3, color=0xFF4444)
    gui.show(directory + f'images/{f:06d}.png')


if __name__ == "__main__":
    x.from_numpy(mesh.vertices.astype(np.float32))
    v.fill(0)
    vertices.from_numpy(mesh.faces)
    cc.fill(0)
    boundary_points.from_numpy(np.array(list(boundary_points_)))
    boundary_edges.from_numpy(boundary_edges_)
    compute_restT_and_m()
    gui = ti.GUI("MPM", (1024, 1024), background_color=0x112F41)
    vertices_ = vertices.to_numpy()
    write_image(0)
    total_time = 0
    for f in range(180):
        total_time -= time.time()
        print("==================== Frame: ", f, " ====================")
        initial_guess()
        prs = []
        drs = []
        for step in range(2):
            find_constraints()

            data_mat.fill(0)
            data_rhs.fill(0)
            data_x.fill(0)
            global_step()
            data_x.from_numpy(solve_linear_system(data_mat.to_numpy(), data_rhs.to_numpy(), n_particles * 2, dirichlet, xn.to_numpy(), False, 0, cnt[None]))
            apply_newton_result()

            local_step()

            pr = prime_residual()
            prs.append(math.log(max(pr, 1e-20)))
            dr = dual_residual()
            drs.append(math.log(max(dr, 1e-20)))
            xr = X_residual()
            print(f, "/", step, f" change of X: {xr:.8f}, prime residual: {pr:.8f}, dual residual: {dr:.8f}")

            dual_step()
        print(f, sha1(x.to_numpy()).hexdigest())

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
