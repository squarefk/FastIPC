from reader import *
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
neural = False
neural_str = "_neural" if neural else ""
directory = 'output/' + '_'.join(sys.argv[:2]) + neural_str + '/'
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
sub_steps = 1
n_particles = len(mesh_particles)
n_elements = len(mesh_elements)

x, x0, xPrev, xTilde, xTarget, xn, v, m = vec(), vec(), vec(), vec(), vec(), vec(), vec(), scalar()
ti.root.dense(ti.i, n_particles).place(x, x0, xPrev, xTarget, xTilde, xn, v, m)
zero = vec()
la, mu = scalar(), scalar()
restT = mat()
B = mat()
vertices = ti.field(ti.i32)
ti.root.dense(ti.i, n_particles).place(zero)
ti.root.dense(ti.i, n_elements).place(la, mu)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.i, n_elements).place(B)
ti.root.dense(ti.ij, (n_elements, dim + 1)).place(vertices)

A = ti.Matrix.field(dim * dim, dim * (dim+1), real, n_elements)  # local x->F matrix
Bp = mat()
F = mat()
U = mat()
V = mat()
sigma = vec()
sigma_Bp = vec()
ti.root.dense(ti.i, n_elements).place(F, U, V, sigma)
ti.root.dense(ti.i, 2 * n_elements).place(Bp, sigma_Bp)

MAX_LINEAR = 50000000 if dim == 3 else 5000000
data_rhs = ti.field(real, shape=n_particles * dim)
data_row = ti.field(ti.i32, shape=MAX_LINEAR)
data_col = ti.field(ti.i32, shape=MAX_LINEAR)
data_val = ti.field(real, shape=MAX_LINEAR)
data_sol = ti.field(real, shape=n_particles * dim)
cnt = ti.field(ti.i32, shape=())

factor = None # prefactorize

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

    for i in range(n_elements):
        # Get (Dm)^-1 for this element:
        Dm_inv_i = B[i]
        if ti.static(dim == 2):
            a = Dm_inv_i[0, 0]
            b = Dm_inv_i[0, 1]
            c = Dm_inv_i[1, 0]
            d = Dm_inv_i[1, 1]
            # Construct A_i:
            A[i][0, 0] = -a-c
            A[i][0, 2] = a
            A[i][0, 4] = c
            A[i][1, 0] = -b-d
            A[i][1, 2] = b
            A[i][1, 4] = d
            A[i][2, 1] = -a-c
            A[i][2, 3] = a
            A[i][2, 5] = c
            A[i][3, 1] = -b-d
            A[i][3, 3] = b
            A[i][3, 5] = d
        else:
            print('ERROR, not implemented')


@ti.kernel
def build_lhs():
    cnt[None] = 0
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            c = cnt[None] + i * dim + d
            data_row[c], data_col[c], data_val[c] = i * dim + d, i * dim + d, m[i]
    cnt[None] += dim * n_particles
    if ti.static(dim == 2):
        for ele_idx in range(n_elements):
            A_i = A[ele_idx]
            ia, ib, ic = vertices[ele_idx, 0], vertices[ele_idx, 1], vertices[ele_idx, 2]
            ia_x_idx, ia_y_idx = ia*2, ia*2+1
            ib_x_idx, ib_y_idx = ib*2, ib*2+1
            ic_x_idx, ic_y_idx = ic*2, ic*2+1
            q_idx_vec = ti.Vector([ia_x_idx, ia_y_idx, ib_x_idx, ib_y_idx, ic_x_idx, ic_y_idx])
            vol0 = restT[ele_idx].determinant() / dim / (dim - 1)
            for A_row_idx in ti.static(range(6)):
                for A_col_idx in ti.static(range(6)):
                    lhs_row_idx = q_idx_vec[A_row_idx]
                    lhs_col_idx = q_idx_vec[A_col_idx]
                    c = cnt[None] + ele_idx * 36 + A_row_idx * 6 + A_col_idx
                    data_row[c], data_col[c], data_val[c] = lhs_row_idx, lhs_col_idx, 0.
                    for idx in ti.static(range(4)):
                        data_val[c] += dt * dt * vol0 * (2 * mu[ele_idx] + la[ele_idx]) * (A_i[idx,A_row_idx] * A_i[idx,A_col_idx])
        cnt[None] += n_elements * 36
    else:
        print("Not implemented")
    if cnt[None] >= MAX_LINEAR:
        print("FATAL ERROR: Array Too Small!")
    print("Total entries: ", cnt[None])

gradient_fd = ti.field(real, shape=n_particles * dim)


def prefactorize(current_time):
    build_lhs()
    dirichlet_fixed, _ = settings['dirichlet'](current_time)
    D = np.stack((dirichlet_fixed,) * dim, axis=-1).reshape((n_particles * dim))
    dfx.from_numpy(D.astype(np.int32))
    @ti.kernel
    def DBC_set_zeros():
        for i in range(cnt[None]):
            if dfx[data_row[i]] or dfx[data_col[i]]:
                data_val[i] = 0
    DBC_set_zeros()
    n = n_particles * dim
    row, col, val = data_row.to_numpy()[:cnt[None]], data_col.to_numpy()[:cnt[None]], data_val.to_numpy()[:cnt[None]]
    A = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n))
    D = np.where(D)[0]
    A += scipy.sparse.csr_matrix((np.ones(len(D)), (D, D)), shape=(n, n))
    global factor
    factor = cholesky(A)
    data_row.fill(0)
    data_col.fill(0)
    data_val.fill(0)
    

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
                xTarget(d)[i] = dirichlet_value[i, d]


@ti.kernel
def update_F():
    for e in range(n_elements):
        F[e] = compute_T(e) @ B[e]
        U[e], sig, V[e] = ti.svd(F[e])
        if ti.static(dim == 2):
            sigma[e] = ti.Vector([sig[0,0], sig[1,1]])
        else:
            sigma[e] = ti.Vector([sig[0,0], sig[1,1], sig[2,2]])


@ti.kernel
def project_F_pd():
    for e in range(n_elements):
        Bp[e] = U[e] @ V[e].transpose()
    
        # Construct volume preservation constraints:
        sig = sigma[e]
        x, y, max_it, tol = 10.0, 10.0, 80, 1e-6
        for t in range(max_it):
            aa, bb = x + sig[0], y + sig[1]
            f = aa * bb - 1
            g1, g2 = bb, aa
            bot = g1 * g1 + g2 * g2
            top = x * g1 + y * g2 - f
            div = top / bot
            x0, y0 = x, y
            x = div * g1
            y = div * g2
            dx, dy = x - x0, y - y0
            if dx * dx + dy * dy < tol * tol:
                break
        PP = ti.Matrix.rows([[x + sig[0], 0.0], [0.0, sig[1] + y]])
        Bp[n_elements + e] = U[e] @ PP @ V[e].transpose()


@ti.kernel
def compute_energy() -> real:
    total_energy = 0.0
    # inertia
    for i in range(n_particles):
        total_energy += 0.5 * m[i] * (x[i] - xTilde[i]).norm_sqr()
    # elasticity
    for e in range(n_elements):
        vol0 = restT[e].determinant() / dim / (dim - 1)
        total_energy += (dt * dt) * vol0 * mu[e] * (F[e] - Bp[e]).norm_sqr() + 0.5 * (dt * dt) * vol0 * la[e] * (F[e] - Bp[e + n_elements]).norm_sqr()
    return total_energy


@ti.kernel
def compute_inertia():
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            # data_rhs[i * dim + d] += one_over_dt2 * m[i] * xTilde(d)[i]
            data_rhs[i * dim + d] -= m[i] * (x(d)[i] - xTilde(d)[i])


@ti.kernel
def compute_elasticity():
    for ele_idx in range(n_elements):
        vol0 = restT[ele_idx].determinant() / dim / (dim - 1)
        if ti.static(dim == 2):
            ia, ib, ic = vertices[ele_idx, 0], vertices[ele_idx, 1], vertices[ele_idx, 2]
            Bp_i = Bp[ele_idx]  # It is a 2x2 matrix now. We want it be a 4x1 vector.
            Bp_i_vec_mu = ti.Vector([Bp_i[0, 0], Bp_i[0, 1], Bp_i[1, 0], Bp_i[1, 1]])
            Bp_i = Bp[ele_idx + n_elements]  # It is a 2x2 matrix now. We want it be a 4x1 vector.
            Bp_i_vec_lam = ti.Vector([Bp_i[0, 0], Bp_i[0, 1], Bp_i[1, 0], Bp_i[1, 1]])
            local_x = ti.Vector([x[ia][0], x[ia][1], x[ib][0], x[ib][1], x[ic][0], x[ic][1]])
            # local_target_x = ti.Vector([xTarget[ia][0], xTarget[ia][1], xTarget[ib][0], xTarget[ib][1], xTarget[ic][0], xTarget[ic][1]])
            A_i = A[ele_idx]
            F_vec = A_i @ local_x
            F_minus_Bp_mu = F_vec - Bp_i_vec_mu
            F_minus_Bp_lam = F_vec - Bp_i_vec_lam
            AT_Bp = dt * dt * vol0 * A_i.transpose() @ (2 * mu[ele_idx] * F_minus_Bp_mu + la[ele_idx] * F_minus_Bp_lam)
            # target_x_contribution = vol0 * (2 * mu[ele_idx] + la[ele_idx]) * A_i.transpose() @ (A_i @ local_target_x)
            # AT_Bp -= target_x_contribution # dirichlet projection
        
            # Add AT_Bp back to rhs
            q_ia_x_idx = ia*2
            q_ia_y_idx = q_ia_x_idx+1
            data_rhs[q_ia_x_idx] -= AT_Bp[0]
            data_rhs[q_ia_y_idx] -= AT_Bp[1]

            q_ib_x_idx = ib*2
            q_ib_y_idx = q_ib_x_idx+1
            data_rhs[q_ib_x_idx] -= AT_Bp[2]
            data_rhs[q_ib_y_idx] -= AT_Bp[3]

            q_ic_x_idx = ic*2
            q_ic_y_idx = q_ic_x_idx+1
            data_rhs[q_ic_x_idx] -= AT_Bp[4]
            data_rhs[q_ic_y_idx] -= AT_Bp[5]
        else:
            print("Not implemented!!")
        
    # for i in range(NV):
    #     for d in ti.static(range(dim)):
    #         if dfx[i * dim + d]:
    #             rhs[i * dim + d] = xTarget[i][d]


def build_rhs():
    data_rhs.fill(0)
    compute_inertia()
    compute_elasticity()

            
def solve_system(current_time):
    dirichlet_fixed, dirichlet_value = settings['dirichlet'](current_time)
    D = np.stack((dirichlet_fixed,) * dim, axis=-1).reshape((n_particles * dim))
    D = np.where(D)[0]
    rhs = data_rhs.to_numpy()
    rhs[D] = 0
    sol = factor(rhs)
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
        prefactorize(0)
        save_x0()
        zero.fill(0)
        write_image(0)
        f_start = 0
        if len(sys.argv) == 3:
            f_start = int(sys.argv[2])
            [x_, v_] = pickle.load(open(directory + f'caches/{f_start:06d}.p', 'rb'))
            x.from_numpy(x_)
            v.from_numpy(v_)
        pd_iter_total = 0
        for f in range(f_start, 10000):
            with Timer("Time Step"):
                print("==================== Frame: ", f, " ====================")
                for step in range(sub_steps):
                    print("============== Substep: ", step, " ==============")
                    compute_xn_and_xTilde()
                    move_nodes(f * dt)
                    pd_iter = 0
                    while True:
                        pd_iter += 1
                        print("-------------------- PD Iteration: ", pd_iter, " --------------------")
                        with Timer("Build System"):
                            update_F()
                            project_F_pd()
                            build_rhs()
                        with Timer("Solve System"):
                            solve_system(f * dt)
                            save_xPrev()
                            apply_sol(1.0)
                        if output_residual() < 1e-2:
                            break
                    compute_v()
                    pd_iter_total += pd_iter
                print("Avg PD iter: ", pd_iter_total / (f + 1))
            with Timer("Visualization"):
                write_image(f + 1)
            pickle.dump([x.to_numpy(), v.to_numpy()], open(directory + f'caches/{f + 1:06d}.p', 'wb'))
            Timer_Print()
