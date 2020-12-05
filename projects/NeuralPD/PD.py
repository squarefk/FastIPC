##################### Load Neural Network ######################
# import sys
# import os
# import tensorflow as tf
# import numpy as np
# import tensorflow_probability as tfp
# from math import hypot

# os.system("scp -r xuan@jg3:/home/xuan/code/PINN-research/PD/min_psi .")
# # os.system("scp -r xuan@jg3:/home/xuan/code/PINN-research/PD/family_manifold .")

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

model = "fc_dpsi_linesearch"
# descending_sigma = True
neural = True
# # mu_model = tf.keras.models.load_model("min_psi/nk_polar/mu".format(model), custom_objects={'tf': tf})
# mu_model = tf.keras.models.load_model("min_psi/{}/mu".format(model), custom_objects={'tf': tf})
# lam_model = tf.keras.models.load_model("min_psi/{}/lam".format(model), custom_objects={'tf': tf})
# mu_model.summary()
# lam_model.summary()

mu_scaling = 1
lam_scaling = 100.
# lam_model = tf.keras.models.load_model("min_psi/nk_polar/lam".format(model), custom_objects={'tf': tf})

# mu_model = tf.keras.models.load_model("svd_pd_fc/mu".format(surfix), custom_objects={'tf': tf})
# lam_model = tf.keras.models.load_model("svd_pd_fc/lam".format(surfix), custom_objects={'tf': tf})



############################ Taichi ####################

from reader import *
from common.math.math_tools import *
from common.utils.timer import *
from common.utils.logger import *
from physics.fixed_corotated import *

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
neural_str = f"_neural_{model}" if neural else ""
directory = 'output/' + '_'.join(sys.argv[:2]) + neural_str + '/'
settings['directory'] = directory
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
# Bp_prev = mat()
F = mat()
U = mat()
V = mat()
sigma = vec()
sigma_Bp = vec()
sigma_Bp_Prev = vec()
sigma_Bp_change = vec()
energy_change = scalar()
radius = scalar()
# local_energy_change = scalar()
ti.root.dense(ti.i, n_elements).place(F, U, V, sigma)
ti.root.dense(ti.i, 2 * n_elements).place(Bp, sigma_Bp, sigma_Bp_Prev, radius, energy_change, sigma_Bp_change)

MAX_LINEAR = 50000000 if dim == 3 else 5000000
data_rhs = ti.field(real, shape=n_particles * dim)
data_row = ti.field(ti.i32, shape=MAX_LINEAR)
data_col = ti.field(ti.i32, shape=MAX_LINEAR)
data_val = ti.field(real, shape=MAX_LINEAR)
data_sol = ti.field(real, shape=n_particles * dim)
cnt = ti.field(ti.i32, shape=())
inertia_energy = ti.field(real, shape=())
elastic_energy = ti.field(real, shape=())
elastic_energy_gt = ti.field(real, shape=())

factor = None # prefactorize

show_ground_truth = False

dfx = ti.field(ti.i32, shape=n_particles * dim)
last_D = np.array([True] * n_particles)

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
    return  E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


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
        sigma_Bp[i] = ti.Vector([1., 1.])
        sigma_Bp[i + n_elements] = ti.Vector([1., 1.])
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
                        data_val[c] += dt * dt * vol0 * (2 * mu_scaling * mu[ele_idx] + lam_scaling * la[ele_idx]) * (A_i[idx,A_row_idx] * A_i[idx,A_col_idx])
        cnt[None] += n_elements * 36
    else:
        print("Not implemented")
    if cnt[None] >= MAX_LINEAR:
        print("FATAL ERROR: Array Too Small!")
    print("Total entries: ", cnt[None])


def factorize(current_time):
    dirichlet_fixed, _ = settings['dirichlet'](current_time)
    global factor, last_D
    D = np.stack((dirichlet_fixed,) * dim, axis=-1).reshape((n_particles * dim))
    if (dirichlet_fixed == last_D).all():
        return
    last_D = dirichlet_fixed
    build_lhs()
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

@ti.func
def psi(theta, r, s1, s2):
    return ((s1 + r * ti.cos(theta)) * (s2 + r * ti.sin(theta)) - 1) ** 2
@ti.func
def dpsi(theta, r, s1, s2):
    return 2 * r * (r * ti.cos(2 * theta) + s1 * ti.cos(theta) - s2 * ti.sin(theta)) * ((s1 + r * ti.cos(theta)) * (s2 + r * ti.sin(theta)) - 1)
@ti.func
def ddpsi(theta, r, s1, s2):
    return 2 * r * (r * (r * ti.cos(2 * theta) + s1 * ti.cos(theta) - s2 * ti.sin(theta)) ** 2 - (r * s1 * ti.sin(theta) + r * ti.cos(theta) * (r * ti.sin(theta) + s2) + s1 * s2 - 1) * (ti.cos(theta) * (4 * r * ti.sin(theta) + s2) + s1 * ti.sin(theta)))


@ti.func
def find_lowest_potential(r, s1, s2):
    theta = 3.14
    for _ in range(1000):
        theta_prev = theta
        E0 = psi(theta, r, s1, s2)
        ng = -dpsi(theta, r, s1, s2)
        h = ti.abs(ddpsi(theta, r, s1, s2))
        dtheta = ng / h
        if ti.abs(dtheta) < 1e-3:
            break
        alpha = 1.
        theta = theta_prev + alpha * dtheta
        E = psi(theta, r, s1, s2)
        while E > E0:
            alpha *= 0.5
            theta = theta_prev + alpha * dtheta
            E = psi(theta, r, s1, s2)
    x = s1 + r * ti.cos(theta)
    y = s2 + r * ti.sin(theta)
    return ti.Vector([x, y])


@ti.kernel
def proejct_F_dpsi():
    for e in range(n_elements):
        Bp[e] = U[e] @ V[e].transpose()
        # Construct volume preservation constraints:.
        sig = sigma[e]
        s1, s2 = sig[0], sig[1]
        
        sigmaProdm1lambda = 2. * (s1 * s2 - 1)
        dpsi = ti.Vector([s2, s1]) * sigmaProdm1lambda
        # r = ti.abs((s1 * s2 - 1)) / ti.sqrt(lam_scaling)
        sigma_Bp[e + n_elements] = sig - 0.5 * dpsi / lam_scaling

    for e in range(n_elements):
        PP = ti.Matrix.rows([[sigma_Bp[e+n_elements][0], 0.0], [0.0, sigma_Bp[e+n_elements][1]]])
        old_energy = (F[e] - Bp[e + n_elements]).norm_sqr()
        Bp[e + n_elements] = U[e] @ PP @ V[e].transpose()
        new_energy = (F[e] - Bp[e + n_elements]).norm_sqr()
        energy_change[e + n_elements] = new_energy - old_energy


@ti.kernel
def proejct_F_dpsi_linesearch():
    E0 = 0.
    E = 0.
    succeed = True
    for e in range(n_elements):
        Bp[e] = U[e] @ V[e].transpose()
        # Construct volume preservation constraints:.
        sig = sigma[e]
        s1, s2 = sig[0], sig[1]
        sigmaProdm1lambda = 2. * (s1 * s2 - 1)
        dpsi = ti.Vector([s2, s1]) * sigmaProdm1lambda
        # r = ti.abs((s1 * s2 - 1)) / ti.sqrt(lam_scaling)
        E0 += (sig - sigma_Bp[e + n_elements]).norm_sqr()
        sigma_Bp[e + n_elements] = sig - 0.5 * dpsi / lam_scaling
        E += (sig - sigma_Bp[e + n_elements]).norm_sqr()
    

    if E > E0:
        succeed = False
        while E > E0:
            for e in range(n_elements):
                E -= (sig - sigma_Bp[e + n_elements]).norm_sqr()
                sigma_Bp[e + n_elements] = sig + 0.5 * (sigma_Bp[e + n_elements] - sig)
                E += (sig - sigma_Bp[e + n_elements]).norm_sqr()


    for e in range(n_elements):
        PP = ti.Matrix.rows([[sigma_Bp[e+n_elements][0], 0.0], [0.0, sigma_Bp[e+n_elements][1]]])
        old_energy = (F[e] - Bp[e + n_elements]).norm_sqr()
        Bp[e + n_elements] = U[e] @ PP @ V[e].transpose()
        new_energy = (F[e] - Bp[e + n_elements]).norm_sqr()
        energy_change[e + n_elements] = new_energy - old_energy
    
    return succeed


@ti.kernel
def proejct_F_dpsi_normalized():
    for e in range(n_elements):
        Bp[e] = U[e] @ V[e].transpose()
        # Construct volume preservation constraints:.
        sig = sigma[e]
        s1, s2 = sig[0], sig[1]
        
        sigmaProdm1lambda = 2. * (s1 * s2 - 1)
        dpsi = ti.Vector([s2, s1]) * sigmaProdm1lambda
        r = ti.abs((s1 * s2 - 1)) / ti.sqrt(lam_scaling)
        if dpsi.norm() > 1e-10:
            sigma_Bp[e + n_elements] = sig - dpsi / dpsi.norm() * r
        else:
            sigma_Bp[e + n_elements] = sig
        # r = ti.abs((s1 * s2 - 1)) / ti.sqrt(lam_scaling)
        

    for e in range(n_elements):
        PP = ti.Matrix.rows([[sigma_Bp[e+n_elements][0], 0.0], [0.0, sigma_Bp[e+n_elements][1]]])
        old_energy = (F[e] - Bp[e + n_elements]).norm_sqr()
        Bp[e + n_elements] = U[e] @ PP @ V[e].transpose()
        new_energy = (F[e] - Bp[e + n_elements]).norm_sqr()
        energy_change[e + n_elements] = new_energy - old_energy


@ti.kernel
def project_F_restshape():
    # compute new radius
    for e in range(n_elements):
        Bp[e] = U[e] @ V[e].transpose()
        # Construct volume preservation constraints:.
        sig = sigma[e]
        s1, s2 = sig[0], sig[1]
        direction = ti.Vector([1. - s1, 1. - s2])
        dnorm = direction.norm()
        if dnorm > 1e-10:
            direction /= dnorm
        r = ti.abs((s1 * s2 - 1)) / ti.sqrt(lam_scaling)
        sigma_Bp[e+n_elements] = sig + r * direction

    for e in range(n_elements):
        PP = ti.Matrix.rows([[sigma_Bp[e+n_elements][0], 0.0], [0.0, sigma_Bp[e+n_elements][1]]])
        old_energy = (F[e] - Bp[e + n_elements]).norm_sqr()
        Bp[e + n_elements] = U[e] @ PP @ V[e].transpose()
        new_energy = (F[e] - Bp[e + n_elements]).norm_sqr()
        energy_change[e + n_elements] = new_energy - old_energy


@ti.kernel
def project_F_fc_naive():
    # compute new radius
    E0 = 0.
    E = 0.
    for e in range(n_elements):
        Bp[e] = U[e] @ V[e].transpose()
        # Construct volume preservation constraints:.
        sig = sigma[e]
        s1, s2 = sig[0], sig[1]
        r = ti.abs((s1 * s2 - 1)) / ti.sqrt(lam_scaling)
        if r > 1e-10:
            s1, s2 = sig[0], sig[1]
            new_sig = find_lowest_potential(r, s1, s2)
            sigma_Bp[e+n_elements] = new_sig
        else:
            sigma_Bp[e+n_elements] = sig

    for e in range(n_elements):
        PP = ti.Matrix.rows([[sigma_Bp[e+n_elements][0], 0.0], [0.0, sigma_Bp[e+n_elements][1]]])
        old_energy = (F[e] - Bp[e + n_elements]).norm_sqr()
        Bp[e + n_elements] = U[e] @ PP @ V[e].transpose()
        new_energy = (F[e] - Bp[e + n_elements]).norm_sqr()
        energy_change[e + n_elements] = new_energy - old_energy


@ti.kernel
def project_F_fc_linesearch():
    # backup
    for e in range(n_elements * 2):
        sigma_Bp_Prev[e] = sigma_Bp[e]
    
    # compute dsigma
    E0 = 0.
    E = 0.
    for e in range(n_elements):
        Bp[e] = U[e] @ V[e].transpose()
        # Construct volume preservation constraints:.
        sig = sigma[e]
        s1, s2 = sig[0], sig[1]
        r = ti.abs((s1 * s2 - 1)) / ti.sqrt(lam_scaling)
        sigma_Bp_n = sigma_Bp_Prev[e + n_elements]
        E0 += (sig - sigma_Bp_n).norm_sqr()
        E += (s1 * s2 - 1) ** 2
        if r < 1e-6:
            sigma_Bp_change[e+n_elements] = ti.Vector([0., 0.])
            sigma_Bp[e+n_elements] = sigma_Bp_n
        else:
            new_sig = find_lowest_potential(r, s1, s2)
            sigma_Bp_change[e+n_elements] = new_sig - sigma_Bp_n
            sigma_Bp[e+n_elements] = new_sig
    
    # linesearch
    alpha = 1.0 
    while E > E0:
        E = 0.
        alpha *= 0.5
        for e in range(n_elements):
            sig = sigma[e + n_elements]
            sigma_Bp_n = sigma_Bp_Prev[e + n_elements]
            new_sig = sigma_Bp[e+n_elements] 
            if (sig - sigma_Bp_n).norm_sqr() < (sig - sigma_Bp[e+n_elements]).norm_sqr():
                new_sig = sigma_Bp_n + alpha * sigma_Bp_change[e+n_elements]
            E += (sig - new_sig).norm_sqr()
        if alpha < 1e-6:
            alpha = 1.0
            print("line search stuck!!")
            break

        
    for e in range(n_elements):
        sig = sigma[e + n_elements]
        sigma_Bp_n = sigma_Bp_Prev[e + n_elements]
        new_sig = sigma_Bp[e+n_elements]
        if (sig - sigma_Bp_n).norm_sqr() < (sig - sigma_Bp[e+n_elements]).norm_sqr():
            new_sig = sigma_Bp_n + alpha * sigma_Bp_change[e+n_elements]
            sigma_Bp[e+n_elements] = new_sig
    print(E, E0)        
    
    for e in range(n_elements):
        PP = ti.Matrix.rows([[sigma_Bp[e+n_elements][0], 0.0], [0.0, sigma_Bp[e+n_elements][1]]])
        Bp[e + n_elements] = U[e] @ PP @ V[e].transpose()


@ti.kernel
def project_F_fc():

    # compute new radius
    E0 = 0.
    E = 0.
    for e in range(n_elements):
        Bp[e] = U[e] @ V[e].transpose()
        # Construct volume preservation constraints:.
        sig = sigma[e]
        s1, s2 = sig[0], sig[1]
        r = ti.abs((s1 * s2 - 1)) / ti.sqrt(lam_scaling)
        sigma_Bp_n = sigma_Bp[e + n_elements]
        E0 += (sig - sigma_Bp_n).norm_sqr()
        E += r * r
        radius[e + n_elements] = r
    # linesearch
    while E > E0 + 1e-8:
        for e in range(n_elements):
            sig = sigma[e]
            sigma_Bp_n = sigma_Bp[e + n_elements]
            if (sig - sigma_Bp_n).norm() < radius[e + n_elements]:
                E -= 0.75 * radius[e + n_elements] ** 2
                radius[e + n_elements] *= 0.5
        print(E, E0)
    # update Bp
    for e in range(n_elements):
        if radius[e + n_elements] > 1e-10:
            sig = sigma[e]
            s1, s2 = sig[0], sig[1]
            new_sig = find_lowest_potential(radius[e + n_elements], s1, s2)
            sigma_Bp[e+n_elements] = new_sig
        else:
            sigma_Bp[e+n_elements] = sigma[e]

    for e in range(n_elements):
        PP = ti.Matrix.rows([[sigma_Bp[e+n_elements][0], 0.0], [0.0, sigma_Bp[e+n_elements][1]]])
        Bp[e + n_elements] = U[e] @ PP @ V[e].transpose()


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
def construct_Bp():
    for i in range(n_elements):
        Bp[i] = U[i] @ ti.Matrix.rows([[sigma_Bp[i][0], 0], 
                                        [0, sigma_Bp[i][1]]]) @ V[i].transpose()
        Bp[i+n_elements] = U[i] @ ti.Matrix.rows([[sigma_Bp[i+n_elements][0],0], 
                                                 [0, sigma_Bp[i+n_elements][1]]]) @ V[i].transpose()


def project_F_neural():
    if not descending_sigma:
        sigma_tf = tf.convert_to_tensor(sigma.to_numpy(), dtype=tf.float32)[:, ::-1]
        s_mu = mu_model(sigma_tf)[:, ::-1]
        s_lam = lam_model(sigma_tf)[:, ::-1]
    else:
        sigma_tf = tf.convert_to_tensor(sigma.to_numpy(), dtype=tf.float32)
        s_mu = mu_model(sigma_tf)
        s_lam = lam_model(sigma_tf)
    sigma_Bp.from_numpy(tf.concat([s_mu, s_lam], axis=0).numpy().astype(np.float64))
    # sigma_Bp.from_numpy(tf.concat([s_mu, s_lam], axis=0).numpy())
    construct_Bp()


@ti.kernel
def compute_energy() -> real:
    total_energy = 0.0
    # inertia
    for i in range(n_particles):
        total_energy += 0.5 * m[i] * (x[i] - xTilde[i]).norm_sqr()
    inertia_energy[None] = total_energy
    # elasticity
    elastic_energy_gt[None] = 0
    for e in range(n_elements):
        vol0 = restT[e].determinant() / dim / (dim - 1)
        total_energy += (dt * dt) * vol0 * mu_scaling * mu[e] * (F[e] - Bp[e]).norm_sqr() + 0.5 * (dt * dt) * vol0 * lam_scaling * la[e] * (F[e] - Bp[e + n_elements]).norm_sqr()
        elastic_energy_gt[None] +=  vol0 * elasticity_energy(sigma[e], la[e], mu[e])
    elastic_energy[None] = (total_energy - inertia_energy[None]) / (dt * dt)
    return total_energy

@ti.kernel
def compute_energy_fc() -> real:
    total_energy = 0.0
    # inertia
    for i in range(n_particles):
        total_energy += 0.5 * m[i] * (x[i] - xTilde[i]).norm_sqr()
    # elasticity
    for e in range(n_elements):
        F = compute_T(e) @ B[e]
        vol0 = restT[e].determinant() / dim / (dim - 1)
        U, sig, V = ti.svd(F)
        total_energy += elasticity_energy(sig, la[e], mu[e]) * dt * dt * vol0
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
            AT_Bp = dt * dt * vol0 * A_i.transpose() @ (2 * mu_scaling * mu[ele_idx] * F_minus_Bp_mu + lam_scaling * la[ele_idx] * F_minus_Bp_lam)
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


def build_rhs():
    data_rhs.fill(0)
    compute_inertia()
    compute_elasticity()

            
def solve_system(current_time):
    factorize(current_time)
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
    return residual


if dim == 2:
    gui = ti.GUI("PD", (768, 768), background_color=0x112F41)
else:
    scene = t3.Scene()
    model = t3.Model(f_n=n_boundary_triangles, vi_n=n_particles)
    scene.add_model(model)
    camera = t3.Camera((768, 768))
    scene.add_camera(camera)
    light = t3.Light([0.4, -1.5, 1.8])
    scene.add_light(light)
    gui = ti.GUI('PD', camera.res)

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
        current_time = 0
        for f in range(f_start, 10000):
            with Timer("Time Step"):
                print("==================== Frame: ", f, " ====================")
                for step in range(sub_steps):
                    print("============== Substep: ", step, " ==============")
                    compute_xn_and_xTilde()
                    move_nodes(current_time)
                    pd_iter = 0
                    while True:
                        pd_iter += 1
                        print("-------------------- PD Iteration: ", pd_iter, " --------------------")
                        with Timer("Build System"):
                            update_F()
                            print(f"sigma.max: {sigma.to_numpy().max():.4f}, sigma.min: {sigma.to_numpy().min():.4f}")
                            # if neural:
                            #     project_F_neural()
                            # else:
                            #     project_F_pd()
                            # if pd_iter % 2 == 1:
                            energy_change.fill(0)
                            # project_F_fc_naive()
                            # proejct_F_dpsi_normalized()
                            # proejct_F_dpsi()
                            succeed = proejct_F_dpsi_linesearch()
                            eg = energy_change.to_numpy()
                            print(f"element energy change mean: {eg.mean():.6f}, element energy change min: {eg.min():.6f}, element energy change max: {eg.max():.6f}, element energy change sum: {eg.sum():.6f}")
                            # project_F_fc_linesearch()
                            # else:
                            #     project_F_fc()
                            compute_energy()
                            print(f"inertial energy: {inertia_energy[None]:.4f}, elastic energy: {elastic_energy[None]:.4f}, elastic_energy_gt: {elastic_energy_gt[None]:.4f}")
                            build_rhs()
                        with Timer("Solve System"):
                            solve_system(current_time)
                            save_xPrev()
                            E0 = compute_energy_fc()
                            alpha = 1.
                            apply_sol(alpha)
                            # update_F()
                            # E = compute_energy_fc()
                            # while E > E0:
                            #     alpha *= 0.5
                            #     apply_sol(alpha)
                            #     update_F()
                            #     E = compute_energy_fc()

                        residual = output_residual()
                        print("Search Direction Residual : ", residual / dt, "Projection success: ", )
                        if pd_iter % 2 == 1 and residual < 1e-4 * dt:
                            break
                        
                    compute_v()
                    current_time += dt
                    pd_iter_total += pd_iter
                    
                print("Avg PD iter: ", pd_iter_total / (f + 1))
            with Timer("Visualization"):
                write_image(f + 1)
            pickle.dump([x.to_numpy(), v.to_numpy()], open(directory + f'caches/{f + 1:06d}.p', 'wb'))
            Timer_Print()
