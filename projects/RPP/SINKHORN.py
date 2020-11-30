
import sys
import taichi as ti
import numpy as np
from numpy import linalg as LA
from common.utils.timer import *
import math

ti.init(arch=ti.cpu, kernel_profiler=True)

screen_res = (1000, 1000)
bg_color = 0x112f41
img = ti.field(ti.f32, shape=(800, 800))

# sim dimension
dim = 2
# grid number per particle in each dimension
dpp = 4

# grid data
ni = 100
nj = 100
Ng = ni * nj
Nl = (ni + 1) + (nj + 1)
dx = 1.0 / ni
inv_dx  = float(ni)
lines_begin = ti.Vector.field(dim, float, Nl)
lines_end = ti.Vector.field(dim, float, Nl)
triangles_a = ti.Vector.field(dim, float, 2*Ng)
triangles_b = ti.Vector.field(dim, float, 2*Ng)
triangles_c = ti.Vector.field(dim, float, 2*Ng)
triangles_color = ti.field(float, 2*Ng)
triangles_color_rgb = ti.Vector.field(3, float, 2*Ng)
grid_pos = ti.Vector.field(dim, float, Ng)

# particle data
np_i = ni // dpp
np_j = nj // dpp
Np = np_i * np_j
particle_pos = ti.Vector.field(dim, float, Np)
centroid_pos = ti.Vector.field(dim, float, Np)
particle_color = ti.field(int, Np)

# sinkhorn parameters
domain_volume = 1.0
max_iters = 5000   
eps = 4*dx*dx
ww = 3*ti.ceil(ti.sqrt(eps) / dx)
vol_p = domain_volume / Np
vol_g = domain_volume / Ng
# a = ti.field(float, Np)
a = ti.field(float)
# b = ti.field(float, Ng)
b = ti.field(float)
u = ti.field(float, Np)
# v = ti.field(float, Ng)
v = ti.field(float)
ti.root.dense(ti.ij, (ni, nj)).dense(ti.ij, (2*ww, 2*ww)).place(v)
# KV = ti.field(float, Np)
KV = ti.field(float)
# KtU = ti.field(float, Ng)
KtU = ti.field(float)
Trowsum = ti.field(float, Np)
Tcolsum = ti.field(float, Ng)
basis_sum = ti.field(float, Ng)

# specify data layout
ti.root.dense(ti.i, Np).place(a, KV)
ti.root.dense(ti.i, Ng).place(b, KtU)
print("Particle volume: ", vol_p, ", grid volume: ", vol_g)

@ti.kernel
def init_particles():
    offs = ti.Vector([1, 1]) * dx * dpp * 0.5
    for i, j in ti.ndrange(np_i, np_j):
        k = i * np_i + j
        # particle_pos[k] = ti.Vector([i, j]) * dx * dpp + offs
        particle_pos[k] = ti.Vector([i, j]) * dx * dpp + offs + ti.Vector([ti.random(float), ti.random(float)]) * dx
        particle_color[k] = int(ti.random(float) * 0xffffff)
        a[k] = vol_p
        u[k] = 1.0

@ti.kernel
def init_grid():
    offs = ti.Vector([1, 1]) * dx * 0.5
    for i, j in ti.ndrange(ni, nj):
        k = i * ni + j
        grid_pos[k] = ti.Vector([i, j]) * dx + offs
        triangles_a[2*k] = ti.Vector([i, j]) * dx
        triangles_b[2*k] = ti.Vector([i + 1, j]) * dx
        triangles_c[2*k] = ti.Vector([i, j + 1]) * dx
        triangles_a[2*k+1] = ti.Vector([i + 1, j]) * dx
        triangles_b[2*k+1] = ti.Vector([i + 1, j + 1]) * dx
        triangles_c[2*k+1] = ti.Vector([i, j + 1]) * dx
        b[k] = vol_g
        # v[k] = 1.0
        v[i,j] = 1.0
    for i in range(ni + 1):
        lines_begin[i] = ti.Vector([i * dx, 0]) 
        lines_end[i] = ti.Vector([i * dx, 1])
    for j in range(nj + 1):
        lines_begin[ni + 1 + j] = ti.Vector([0, j * dx]) 
        lines_end[ni + 1 + j] = ti.Vector([1, j * dx]) 

@ti.func
def KijVj():
    # clear data
    shift = ti.Vector([1, 1]) * dx * 0.5
    for i in range(Np):
        KV[i] = 0
    for ii in range(Np):
        pi = particle_pos[ii]
        base = (pi * inv_dx).cast(int)
        for i,j in ti.ndrange((-ww, ww), (-ww, ww)):
            offs = ti.Vector([i,j])
            index = base + offs
            if 0 <= index[0] < ni and 0 <= index[1] < nj:
                # TODO: unitfy the use of index and k here
                k = index[0] * ni + index[1]
                gj = index.cast(float) * dx + shift
                Cij = (pi - gj).norm_sqr()
                Kij = ti.exp(-Cij / eps)
                # ti.atomic_add(KV[ii], Kij*v[k])
                ti.atomic_add(KV[ii], Kij*v[index])

@ti.func
def KVi(ii):
    result = 0.0
    shift = ti.Vector([1, 1]) * dx * 0.5
    pi = particle_pos[ii]
    base = (pi * inv_dx).cast(int)
    for i,j in ti.ndrange((-ww, ww), (-ww, ww)):
        offs = ti.Vector([i,j])
        index = base + offs
        if 0 <= index[0] < ni and 0 <= index[1] < nj:
            # TODO: unitfy the use of index and k here
            k = index[0] * ni + index[1]
            gj = index.cast(float) * dx + shift
            Cij = (pi - gj).norm_sqr()
            Kij = ti.exp(-Cij / eps)
            # TODO: this is slow, is it because of atomic?
            ti.atomic_add(result, Kij*v[index])
    return result


@ti.func
def KjiUi():
    shift = ti.Vector([1, 1]) * dx * 0.5
    # clear data
    for j in range(Ng):
        KtU[j] = 0
    for ii in range(Np):
        pi = particle_pos[ii]
        base = (pi * inv_dx).cast(int)
        for i,j in ti.ndrange((-ww, ww), (-ww, ww)):
            offs = ti.Vector([i,j])
            index = base + offs
            if 0 <= index[0] < ni and 0 <= index[1] < nj:
                # TODO: unitfy the use of index and k here
                k = index[0] * ni + index[1]
                gj = index.cast(float) * dx + shift
                Cij = (pi - gj).norm_sqr()
                Kij = ti.exp(-Cij / eps)
                ti.atomic_add(KtU[k], Kij*u[ii])

@ti.func
def Tij(i,j, index):
    shift = ti.Vector([1, 1]) * dx * 0.5
    pi = particle_pos[i]
    gj = index.cast(float) * dx + shift
    Cij = (pi - gj).norm_sqr()
    Kij = ti.exp(-Cij / eps)
    return float(u[i]*Kij*v[index])

@ti.func
def sumiTij():
    for j in range(Ng):
        Tcolsum[j] = 0
    for ii in range(Np):
        pi = particle_pos[ii]
        base = (pi * inv_dx).cast(int)
        for i,j in ti.ndrange((-ww, ww), (-ww, ww)):
            offs = ti.Vector([i,j])
            index = base + offs
            if 0 <= index[0] < ni and 0 <= index[1] < nj:
                k = index[0] * ni + index[1]
                ti.atomic_add(Tcolsum[k], Tij(ii, k, index))

@ti.func
def sumjTij():
    for i in range(Np):
        Trowsum[i] = 0
    for ii in range(Np):
        pi = particle_pos[ii]
        base = (pi * inv_dx).cast(int)
        for i,j in ti.ndrange((-ww, ww), (-ww, ww)):
            offs = ti.Vector([i,j])
            index = base + offs
            if 0 <= index[0] < ni and 0 <= index[1] < nj:
                k = index[0] * ni + index[1]
                ti.atomic_add(Trowsum[ii], Tij(ii, k, index))

@ti.kernel
def update_centroid():
    shift = ti.Vector([1, 1]) * dx * 0.5
    for i in range(Np):
        centroid_pos[i] = ti.Vector([0.0, 0.0])
    for i in range(2*Ng):
        triangles_color_rgb[i] = ti.Vector([0.0, 0.0, 0.0])
    for ii in range(Np):
        pi = particle_pos[ii]
        base = (pi * inv_dx).cast(int)
        for i,j in ti.ndrange((-ww, ww), (-ww, ww)):
            offs = ti.Vector([i,j])
            index = base + offs
            if 0 <= index[0] < ni and 0 <= index[1] < nj:
                k = index[0] * ni + index[1]
                gj = index.cast(float) * dx + shift
                ti.atomic_add(centroid_pos[ii], Tij(ii, k, index)*gj / Trowsum[ii])
                r,g,b = ti.hex_to_rgb(particle_color[ii])
                rgb_color = ti.Vector([r, g, b])
                ti.atomic_add(triangles_color_rgb[2*k], rgb_color * Tij(ii, k, index) / Tcolsum[k])
                ti.atomic_add(triangles_color_rgb[2*k+1], rgb_color * Tij(ii, k, index) / Tcolsum[k])


@ti.kernel
def sinkhorn_step(iter:int):
    # u step
    # KijVj()
    for i in range(Np):
        u[i] = a[i] / KVi(i)

    # v step
    KjiUi()
    # for j in range(Ng):
        # v[j] = b[j] / KtU[j]
    for i, j in ti.ndrange(ni, nj):
        k = i * ni + j
        v[i,j] = b[k] / KtU[k]

    # if iter % 200 == 0:
    sumiTij()
    sumjTij()



def run_sinkhorn():
    iter = 0
    res_a = LA.norm(a.to_numpy() - Trowsum.to_numpy())
    with Timer("Sinkhorn iteration"):
        while iter < max_iters and res_a > 0.1*vol_p:
        # while iter < max_iters:
            sinkhorn_step(iter)
            res_a = LA.norm(a.to_numpy() - Trowsum.to_numpy())
            res_b = LA.norm(b.to_numpy() - Tcolsum.to_numpy())
            # print(iter, res_a, res_b)
            iter += 1
        print(a)
        print(Trowsum)
    update_centroid()
    for i in range(2*Ng):
        triangles_color[i] = ti.rgb_to_hex(triangles_color_rgb[i])
    print(f'Sinkhorn converges in {iter:.1f}, residual is {res_a:.10f}')

init_particles()
init_grid()
run_sinkhorn()
Timer_Print()
ti.kernel_profiler_print()
gui = ti.GUI('SINKHORN', screen_res)
while gui.running:
    gui.clear(bg_color)
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
    # gui.triangles(triangles_a.to_numpy(), triangles_b.to_numpy(), triangles_c.to_numpy(), triangles_color.to_numpy())
    gui.lines(lines_begin.to_numpy(), lines_end.to_numpy(), radius = 0.6, color=0xff0000)
    gui.circles(particle_pos.to_numpy(), radius=4, color=particle_color.to_numpy())
    gui.circles(centroid_pos.to_numpy(), radius=4, color=0xffffff)
    # gui.circles(grid_pos.to_numpy(), radius=2, color=0xffffff)
    gui.show()