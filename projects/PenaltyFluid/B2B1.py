import taichi as ti
import scipy.sparse
import scipy.sparse.linalg
from sksparse.cholmod import *

ti.init(arch=ti.cpu)

real = ti.f32
dim = 2
MAX_LINEAR = 50000000 if dim == 3 else 5000000

n_particles = 8192
n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho

x = ti.Vector.field(2, real, n_particles)
v = ti.Vector.field(2, real, n_particles)
C = ti.Matrix.field(2, 2, real, n_particles)

data_rhs = ti.field(real, shape=n_particles * dim)
data_row = ti.field(ti.i32, shape=MAX_LINEAR)
data_col = ti.field(ti.i32, shape=MAX_LINEAR)
data_val = ti.field(real, shape=MAX_LINEAR)
data_sol = ti.field(real, shape=n_particles * dim)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

# def solve_system():
#     row = [0, 0, 1, 1]
#     col = [0, 1, 0, 1]
#     val = [1, 2, 2, 1]
#     rhs = [3, 4]
#     A = scipy.sparse.csr_matrix((val, (row, col)), shape=(2, 2))
#     factor = cholesky(A)
#     sol = factor(rhs)
#     print(sol)

@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
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
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
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
    for s in range(50):
        substep()
    gui.clear(0x112F41)
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.show(f'output/{frame:06d}.png')
