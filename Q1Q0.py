import sys
sys.path.insert(0, "../../build")
from JGSL_WATER import *
import taichi as ti

ti.init(arch=ti.gpu)

scale = 1
dim = 2
dt = 1 / 480
dx = 0.008 / scale
boundary = 4.5
n_particles_x = 100 * scale
n_particles_y = 150 * scale
p_mass = 1.0
res = 640
nn = ti.var(dt=ti.i32, shape=())
mm = ti.var(dt=ti.i32, shape=())
kk = ti.var(dt=ti.i32, shape=())
cc = ti.var(dt=ti.i32, shape=())

n_grid = int(1 / dx)
inv_dx = 1 / dx
n_particles = n_particles_x * n_particles_y

real = ti.f32
scalar = lambda: ti.var(dt=real)
index = lambda: ti.var(dt=ti.i32)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

kind = ti.var(ti.i32)
x, v, gradV = vec(), vec(), mat()
grid_v, grid_m = vec(), scalar()
index_v, index_p, index_w = index(), index(), index()
ti.root.dense(ti.ij, n_grid).place(kind)
ti.root.dense(ti.i, n_particles).place(x, v, gradV)
ti.root.dense(ti.ij, n_grid).place(grid_v, grid_m)
ti.root.dense(ti.ij, n_grid).place(index_v, index_p, index_w)

data1 = ti.var(real, shape=1000000)
data2 = ti.var(real, shape=(3, 4000000))
data3 = ti.var(real, shape=1000000)
data4 = ti.var(real, shape=1000000)

color_buffer = ti.Vector(3, real, shape=(res, res))


def sample_particles():
    for i in range(n_particles_x):
        for j in range(n_particles_y):
            idx = i * n_particles_y + j
            x[idx] = [(boundary + 0.25) * dx + 0.5 * i * dx, (boundary + 0.25) * dx + 0.5 * j * dx]
            v[idx] = [0, 0]


@ti.func
def in_boundary(x : ti.i32, y : ti.i32) -> ti.i32:
    ib = 0
    if x < int(boundary) or x >= int(n_grid - boundary) or y < int(boundary) or y >= int(n_grid - boundary):
        ib = 1
    return ib


@ti.kernel
def mark_cells():
    for p in x:
        base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
        kind[base] = max(kind[base], 1)  # S override F
        if in_boundary(base[0] - 1, base[1]) > 0:
            kind[base[0] - 1, base[1]] = 2
        if in_boundary(base[0] + 1, base[1]) > 0:
            kind[base[0] + 1, base[1]] = 2
        if in_boundary(base[0], base[1] - 1) > 0:
            kind[base[0], base[1] - 1] = 2
        if in_boundary(base[0], base[1] + 1) > 0:
            kind[base[0], base[1] + 1] = 2


@ti.kernel
def p2g():
    for p in x:
        base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
        fx = x[p] * inv_dx - 0.5 - ti.cast(base, real)
        w, dw = [1.0 - fx, fx], [[-1.0, -1.0], [1.0, 1.0]]
        affine = p_mass * gradV[p]
        for i in ti.static(range(2)):
            for j in ti.static(range(2)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i](0) * w[j](1)
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            inv_m = 1 / grid_m[i, j]
            grid_v[i, j] = inv_m * grid_v[i, j]
            grid_v(1)[i, j] -= dt * 9.8


@ti.kernel
def build_system():
    nn[None] = 0
    mm[None] = 0
    kk[None] = 0
    for i, j in kind:
        if kind[i, j] == 1:
            grid_m[i, j] += p_mass * 0.25
            grid_m[i + 1, j] += p_mass * 0.25
            grid_m[i, j + 1] += p_mass * 0.25
            grid_m[i + 1, j + 1] += p_mass * 0.25
            index_v[i, j] = -1
            index_v[i + 1, j] = -1
            index_v[i, j + 1] = -1
            index_v[i + 1, j + 1] = -1
            index_p[i, j] = -1
            if kind[i, j + 1] == 2:
                index_w[i, j], index_w[i, j + 1] = -1, -1
            if kind[i, j - 1] == 2:
                index_w[i, j], index_w[i, j - 1] = -1, -1
            if kind[i + 1, j] == 2:
                index_w[i, j], index_w[i + 1, j] = -1, -1
            if kind[i - 1, j] == 2:
                index_w[i, j], index_w[i - 1, j] = -1, -1
    for i, j in grid_m:
        if index_v[i, j] == -1:
            index_v[i, j] = ti.atomic_add(nn[None], 1)
            for d in ti.static(range(2)):
                data1[index_v[i, j] * 2 + d] = dt / grid_m[i, j]
        if index_p[i, j] == -1:
            index_p[i, j] = ti.atomic_add(mm[None], 1)
        if index_w[i, j] == -1:
            index_w[i, j] = ti.atomic_add(kk[None], 1)
    for i, j in grid_m:
        if index_p[i, j] >= 0:
            w, dw = [[0.5, 0.5], [0.5, 0.5]], [[-1.0, -1.0], [1.0, 1.0]]
            cnt = index_p[i, j] * 8
            for di in ti.static(range(2)):
                for dj in ti.static(range(2)):
                    dweight = inv_dx * ti.Vector([dw[di][0] * w[dj][1], w[di][0] * dw[dj][1]])
                    for d in ti.static(range(2)):
                        data2[0, cnt] = index_v[i + di, j + dj] * 2 + d
                        data2[1, cnt] = index_p[i, j]
                        data2[2, cnt] = -dx * dx * 1 * dweight[d]
                        cnt += 1
    # why cc[None] = mm[None] * 8
    cc[None] = 0
    pa = [0, 0, 1, 1]
    pn = [-1, 1, -1, 1]
    po = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    px = [[0, 0], [1, 0], [0, 0], [0, 1]]
    py = [[0, 1], [1, 1], [1, 0], [1, 1]]
    for i, j in grid_m:
        if kind[i, j] == 1:
            for k in ti.static(range(4)):
                if kind[i + po[k][0], j + po[k][1]] == 2:
                    cnt = ti.atomic_add(cc[None], 1)
                    data2[0, mm[None] * 8 + cnt] = index_v[i + px[k][0], j + px[k][1]] * 2 + pa[k]
                    data2[1, mm[None] * 8 + cnt] = mm[None] + index_w[i, j]
                    data2[2, mm[None] * 8 + cnt] = dx * 0.5 * pn[k]
                    cnt = ti.atomic_add(cc[None], 1)
                    data2[0, mm[None] * 8 + cnt] = index_v[i + px[k][0], j + px[k][1]] * 2 + pa[k]
                    data2[1, mm[None] * 8 + cnt] = mm[None] + index_w[i + po[k][0], j + po[k][1]]
                    data2[2, mm[None] * 8 + cnt] = dx * 0.5 * pn[k]
                    cnt = ti.atomic_add(cc[None], 1)
                    data2[0, mm[None] * 8 + cnt] = index_v[i + py[k][0], j + py[k][1]] * 2 + pa[k]
                    data2[1, mm[None] * 8 + cnt] = mm[None] + index_w[i, j]
                    data2[2, mm[None] * 8 + cnt] = dx * 0.5 * pn[k]
                    cnt = ti.atomic_add(cc[None], 1)
                    data2[0, mm[None] * 8 + cnt] = index_v[i + py[k][0], j + py[k][1]] * 2 + pa[k]
                    data2[1, mm[None] * 8 + cnt] = mm[None] + index_w[i + po[k][0], j + po[k][1]]
                    data2[2, mm[None] * 8 + cnt] = dx * 0.5 * pn[k]
    for i, j in index_v:
        if index_v[i, j] >= 0:
            for d in ti.static(range(2)):
                data3[index_v[i, j] * 2 + d] = grid_m[i, j] / dt * grid_v(d)[i, j]


@ti.kernel
def g2p():
    for i, j in index_v:
        if index_v[i, j] >= 0:
            for d in ti.static(range(2)):
                grid_v(d)[i, j] = data4[index_v[i, j] * 2 + d]
    for p in x:
        base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
        fx = x[p] * inv_dx - 0.5 - ti.cast(base, real)
        w, dw = [1.0 - fx, fx], [[-1.0, -1.0], [1.0, 1.0]]
        new_picV = ti.Vector([0.0, 0.0])
        new_gradV = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        for i in ti.static(range(2)):
            for j in ti.static(range(2)):
                g_v = grid_v[base(0) + i, base(1) + j]
                weight = w[i](0) * w[j](1)
                dweight = inv_dx * ti.Vector([dw[i][0] * w[j](1), w[i](0) * dw[j][1]])
                new_picV += weight * g_v
                new_gradV += g_v @ dweight.transpose()
        v[p] = new_picV
        x[p] += dt * v[p]
        gradV[p] = new_gradV
        for d in ti.static(range(2)):
            if x(d)[p] < (boundary - 3) * dx:
                x(d)[p], v(d)[p] = (boundary - 2) * dx, 0
            if x(d)[p] > 1 - (boundary - 3) * dx:
                x(d)[p], v(d)[p] = 1 - (boundary - 2) * dx, 0



gui = ti.GUI("MPM", (640, 640), background_color=0x112F41)


@ti.kernel
def debug():
    # data1[0], data1[1], data1[2] = 10, 20, 30
    # data2[0, 0], data2[1, 0], data2[2, 0] = 0, 1, 3.6
    # data2[0, 1], data2[1, 1], data2[2, 1] = 1, 0, 3.6
    # data3[0], data3[1] = 1, 2
    # nn[None] = 10
    # mm[None] = 10
    # kk[None] = 10
    for i, j in color_buffer:
        for d in ti.static(range(3)):
            color_buffer(d)[i, j] = ti.cast(kind[int(i / res * n_grid), int(j / res * n_grid)], real) / 2


if __name__ == "__main__":
    sample_particles()
    for frame in range(2400):
        kind.fill(0)
        mark_cells()

        grid_v.fill(0)
        grid_m.fill(0)
        p2g()

        grid_m.fill(0)
        index_v.fill(-2)
        index_p.fill(-2)
        index_w.fill(-2)
        data1.fill(0)
        data2.fill(0)
        data3.fill(0)
        data4.fill(0)
        build_system()

        data4.from_numpy(solve_linear_system(data1.to_numpy(), data2.to_numpy(), data3.to_numpy(), nn[None] * 2, mm[None] + kk[None]))
        grid_v.fill(0)
        g2p()

        particle_pos = x.to_numpy()
        gui.circles(particle_pos, radius=1.5, color=0xF2B134)
        gui.line((boundary * dx, boundary * dx), (boundary * dx, 1 - boundary * dx), color=0xFFFFFF, radius=1)
        gui.line((boundary * dx, 1 - boundary * dx), (1 - boundary * dx, 1 - boundary * dx), color=0xFFFFFF, radius=1)
        gui.line((1 - boundary * dx, 1 - boundary * dx), (1 - boundary * dx, boundary * dx), color=0xFFFFFF, radius=1)
        gui.line((1 - boundary * dx, boundary * dx), (boundary * dx, boundary * dx), color=0xFFFFFF, radius=1)
        gui.show(f'output/{frame:06d}.png')