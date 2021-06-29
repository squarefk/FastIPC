import taichi as ti
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from sksparse.cholmod import *

ti.init(arch=ti.cpu)

real = ti.f32
dim = 2
MAX_LINEAR = 5000000

n_particles = 8192
n_grid = 128
dx = 1 / n_grid
inv_dx = n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho

x = ti.Vector.field(2, real, n_particles)
v = ti.Vector.field(2, real, n_particles)
C = ti.Matrix.field(2, 2, real, n_particles)

data_rhs = ti.field(real, MAX_LINEAR)
data_row = ti.field(ti.i32, MAX_LINEAR)
data_col = ti.field(ti.i32, MAX_LINEAR)
data_val = ti.field(real, MAX_LINEAR)
data_sol = ti.field(real, MAX_LINEAR)
data_cnt = ti.field(ti.i32, ())

grid_v = ti.Vector.field(2, real, (n_grid, n_grid))
grid_m = ti.field(real, (n_grid, n_grid))
grid_id = ti.field(ti.i32, (n_grid, n_grid))
grid_cnt = ti.field(ti.i32, ())

cell_id = ti.field(ti.i32, (n_grid, n_grid))
cell_cnt = ti.field(ti.i32, ())

# def solve_system():
#     row = [0, 0, 1, 1]
#     col = [0, 1, 0, 1]
#     val = [1, 2, 2, 1]
#     rhs = [3, 4]
#     A = scipy.sparse.csr_matrix((val, (row, col)), shape=(2, 2))
#     factor = cholesky(A)
#     sol = factor(rhs)
#     print(sol)

# def matrix_product():
#     row = [0, 0]
#     col = [0, 1]
#     val = [1, 2]
#     A = scipy.sparse.csr_matrix((val, (row, col)), shape=(2, 3))
#     B = A.transpose() @ A
#     C = A @ A.transpose()

@ti.kernel
def p2g():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
        grid_id[i, j] = -1
        cell_id[i, j] = -1
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        affine = p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
        for i, j in ti.static(ti.ndrange(2, 2)):
            offset = ti.Vector([i, j])
            cell_id[base + offset] = 0
    grid_cnt[None] = 0
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
            grid_id[i, j] = ti.atomic_add(grid_cnt[None], 1)
    cell_cnt[None] = 0
    for i, j in cell_id:
        if cell_id[i, j] >= 0:
            cell_id[i, j] = ti.atomic_add(cell_cnt[None], 1)


def compute_system():
    @ti.kernel
    def compute_D_kernel():
        for i, j in cell_id:
            if cell_id[i, j] >= 0:
                if cell_id[i, j] >= 0 and cell_id[i + 1, j] >= 0 and cell_id[i, j + 1] >= 0 and cell_id[i + 1, j + 1] >= 0:
                    gaussian_quadrature = [-3 ** 0.5 / 3, 3 ** 0.5 / 3]
                    for p, q in ti.static(ti.ndrange(2, 2)):
                        fxB1 = ti.Vector([0.5 - gaussian_quadrature[p] * 0.5, 0.5 + gaussian_quadrature[q] * 0.5])
                        wB1 = [1.0 - fxB1, fxB1]
                        fxB2 = ti.Vector([1 - gaussian_quadrature[p] * 0.5, 1 + gaussian_quadrature[q] * 0.5])
                        wB2 = [0.5 * (1.5 - fxB2)**2, 0.75 - (fxB2 - 1)**2, 0.5 * (fxB2 - 0.5)**2]
                        dwB2 = [fxB2 - 1.5, -2 * (fxB2 - 1), fxB2 - 0.5]
                        for x1, y1 in ti.static(ti.ndrange(2, 2)):
                            weightB1 = wB1[x1][0] * wB1[y1][1]
                            for x2, y2 in ti.static(ti.ndrange(3, 3)):
                                dweightB2 = [0., 0.]
                                dweightB2[0] = dwB2[x2][0] * wB2[y2][1] * inv_dx
                                dweightB2[1] = wB2[x2][0] * dwB2[y2][1] * inv_dx
                                for d in ti.static(range(dim)):
                                    c_id = cell_id[i + x1, j + y1]
                                    g_id = grid_id[i + x2, j + y2] * dim + d
                                    tmp = ti.atomic_add(data_cnt[None], 1)
                                    data_row[tmp] = c_id
                                    data_col[tmp] = g_id
                                    data_val[tmp] = p_vol * weightB1 * dweightB2[d]
    data_cnt[None] = 0
    compute_D_kernel()
    assert data_cnt[None] < MAX_LINEAR
    row = data_row.to_numpy()[:data_cnt[None]]
    col = data_col.to_numpy()[:data_cnt[None]]
    val = data_val.to_numpy()[:data_cnt[None]]
    D = scipy.sparse.csr_matrix((val, (row, col)), shape=(cell_cnt[None], grid_cnt[None] * dim))
    DTD = D.transpose() @ D
    print(DTD.todense())

    @ti.kernel
    def compute_M_kernel():
        for i, j in cell_id:
            if cell_id[i, j] >= 0:
                if cell_id[i, j] >= 0 and cell_id[i + 1, j] >= 0 and cell_id[i, j + 1] >= 0 and cell_id[i + 1, j + 1] >= 0:
                    gaussian_quadrature = [-3 ** 0.5 / 3, 3 ** 0.5 / 3]
                    for p, q in ti.static(ti.ndrange(2, 2)):
                        fxB2 = ti.Vector([1 - gaussian_quadrature[p] * 0.5, 1 + gaussian_quadrature[q] * 0.5])
                        wB2 = [0.5 * (1.5 - fxB2)**2, 0.75 - (fxB2 - 1)**2, 0.5 * (fxB2 - 0.5)**2]
                        for x2, y2 in ti.static(ti.ndrange(3, 3)):
                            weightB2 = wB2[x2][0] * wB2[y2][1]
                            for d in ti.static(range(dim)):
                                g_id = grid_id[i + x2, j + y2] * dim + d
                                tmp = ti.atomic_add(data_cnt[None], 1)
                                data_row[tmp] = g_id
                                data_col[tmp] = g_id
                                data_val[tmp] = p_mass * weightB2
                                data_rhs[g_id] += p_mass * weightB2 * grid_v[i + x2, j + y2][d]
    data_cnt[None] = 0
    data_rhs.fill(0)
    compute_M_kernel()
    assert data_cnt[None] < MAX_LINEAR
    row = data_row.to_numpy()[:data_cnt[None]]
    col = data_col.to_numpy()[:data_cnt[None]]
    val = data_val.to_numpy()[:data_cnt[None]]
    rhs = data_rhs.to_numpy()[:grid_cnt[None] * dim]
    M = scipy.sparse.csr_matrix((val, (row, col)), shape=(grid_cnt[None] * dim, grid_cnt[None] * dim))

    factor = cholesky(M)
    sol = factor(rhs)
    sol_extend = np.zeros(MAX_LINEAR)
    sol_extend[:grid_cnt[None] * dim] = sol
    data_sol.from_numpy(sol_extend)
    np.set_printoptions(threshold=np.inf)
    print(sol)

    @ti.kernel
    def apply_sol():
        for i, j in grid_id:
            if grid_id[i, j] >= 0:
                grid_v[i, j][0] = data_sol[grid_id[i, j] * 2]
                grid_v[i, j][1] = data_sol[grid_id[i, j] * 2 + 1]
    apply_sol()


@ti.kernel
def g2p():
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(real, 2)
        new_C = ti.Matrix.zero(real, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        C[p] = new_C

@ti.kernel
def init():
    for i in range(n_particles):
        if i < n_particles // 2:
            x[i] = [ti.random() * 0.2 + 0.2, ti.random() * 0.4 + 0.2]
            v[i] = [1, 0]
        else:
            x[i] = [ti.random() * 0.2 + 0.45, ti.random() * 0.4 + 0.2]
            v[i] = [-1, 0]

init()
gui = ti.GUI('MPM88')
for frame in range(100):
    p2g()
    compute_system()
    g2p()
    gui.clear(0x112F41)
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.show(f'output/{frame:06d}.png')
