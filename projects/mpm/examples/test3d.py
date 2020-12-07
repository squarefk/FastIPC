import os
import numpy as np
import taichi as ti

from projects.mpm.engine.mpm_solver_implicit import MPMSolverImplicit
from common.utils.logger import *

ti.init(arch=ti.cpu, default_fp=ti.f64)

test_case = 11


directory = 'output/3d/' + str(test_case) + '/'
os.makedirs(directory, exist_ok=True)
gui = ti.GUI("PNMPM-3D", res=512, background_color=0x112F41)

mpm = MPMSolverImplicit(res=(64, 64, 64), size=10)

if test_case == 1:  # jello drop on ground
    mpm.symplectic = True
    
    mpm.setDXandDT(DX=0.15625,DT=0.0003125)
    mpm.setGravity((0, -100, 0))
    mpm.setLameParameter(E=1e6, nu=0.2)

    mpm.add_cube(min_corner=(2, 1.0, 3), max_corner=(3, 2.0, 6), num_particles=20000, rho=1000)

    mpm.add_surface_collider(point=(0.0, 0.15, 0.0), normal=(0.0, 1.0, 0.0))
    mpm.add_analytic_box(min_corner=(0.0, 0.0, 0.0), max_corner=(1.0, 0.05, 1.0))

if test_case == 10:  # jello drop on ground
    mpm.symplectic = True
    
    mpm.setDXandDT(DX=0.15625,DT=0.0003125)
    mpm.setGravity((0, -100, 0))
    mpm.setLameParameter(E=1e6, nu=0.2)

    mpm.add_cube(min_corner=(2, 2.0, 3), max_corner=(3, 3.0, 6), num_particles=20000, rho=1000)

    # mpm.add_surface_collider(point=(0.0, 0.15, 0.0), normal=(0.0, 1.0, 0.0))
    # mpm.add_analytic_box(min_corner=(0.0, 0.0, 0.0), max_corner=(10.0, 0.5, 10.0))
    mpm.add_analytic_box(min_corner=(2.0, 0.0, 0.0), max_corner=(3.0, 1.0, 10.0), rotation=(0., 0., 0.38268343, 0.92387953))


if test_case == 11:  # physbam snow
    mpm.symplectic = True
    
    # mpm.cfl = 0.05
    mpm.setDXandDT(DX=0.00501253,DT=0.0001)
    mpm.setGravity((0, -2.5, 0))
    mpm.setLameParameter(E=360, nu=0.3)

    mpm.add_cube(min_corner=(0.3, 0.7, 0.4), max_corner=(0.7, 0.9, 0.6), num_particles=800000, rho=3.0)
    # mpm.load_state('data/3d.txt')

    # mpm.add_surface_collider(point=(0.0, 0.15, 0.0), normal=(0.0, 1.0, 0.0))
    # mpm.add_analytic_box(min_corner=(0.0, 0.0, 0.0), max_corner=(10.0, 0.5, 10.0))
    mpm.add_analytic_box(min_corner=(0.4, 0.3, 0.0), max_corner=(0.6, 0.5, 1.0), rotation=(0., 0., 0.38268343, 0.92387953)) # mound
    mpm.add_analytic_box(min_corner=(-1.0, 0.0, -1.0), max_corner=(1.0, 0.1, 1.0))


if test_case == 2:  # jello drop on ground
    mpm.symplectic = False
    mpm.setGravity((0, -100, 0))

    # mpm.add_cube(min_corner=(2, 6, 3), max_corner=(3, 7, 6))
    mpm.add_cube(min_corner=(2, 1.0, 3), max_corner=(3, 2.0, 6))

    mpm.add_surface_collider(point=(0.0, 0.15, 0.0), normal=(0.0, 1.0, 0.0))
    # mpm.add_analytic_box(min_corner=(0.0, 0.0, 0.0), max_corner=(1.0, 0.05, 1.0))


def writeObjFile(f, X):
    fo = open(directory + "frame_" + str(f) + ".obj", "w")
    for i in range(X.shape[0]):
        fo.write("v " + str(X[i][0]) + " " + str(X[i][1]) + " " +
                 str(X[i][2]) + "\n")
    fo.close()


with Logger(directory + f'log.txt'):
    for frame in range(500):
        print("================frame", frame, "=================")
        mpm.step(4e-3)
        colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                          dtype=np.uint32)
        particles = mpm.particle_info()
        np_x = particles['position'] / 10.0

        # simple camera transform
        screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
        screen_y = (np_x[:, 1])

        screen_pos = np.stack([screen_x, screen_y], axis=-1)

        gui.circles(screen_pos, radius=1.5, color=0x068587)
        gui.show(directory + f'{frame:06d}.png')

        writeObjFile(frame, np_x)

ti.print_profile_info()
