import os
import numpy as np
import taichi as ti

from projects.mpm.engine.mpm_solver_implicit import MPMSolverImplicit
from common.utils.logger import *

ti.init(arch=ti.cpu, default_fp=ti.f64)
# ti.init(arch=ti.gpu, default_fp=ti.f64)

directory = 'output/2d/'
os.makedirs(directory, exist_ok=True)


gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

mpm = MPMSolverImplicit(res=(128,128))


# mpm.load_state()

test_case = 1

if test_case == 1: # physbam snow
    mpm.symplectic = False
    mpm.cfl = 0.6
    mpm.setDXandDT(DX=0.00502513,DT=0.004)
    # mpm.symplectic = True
    # mpm.setDXandDT(DX=0.00502513,DT=0.0005)

    mpm.setGravity((0, -2.0))
    mpm.setLameParameter(E=40,nu=0.2)
    mpm.add_cube(min_corner=(0.3, 0.7), max_corner=(0.7, 0.9), num_particles = 20000)
    # mpm.add_cube(min_corner=(0.3, 0.55), max_corner=(0.7, 0.75), num_particles = 20000)
    mpm.add_analytic_box(min_corner=(0.4, 0.3), max_corner=(0.6, 0.5), rotation=3.1415926/4)
    mpm.add_analytic_box(min_corner=(0.0, 0.0), max_corner=(1.0, 0.05))
    mpm.load_state()

elif test_case == 2: # jello drop
    mpm.add_cube(min_corner=(0.3, 0.55), max_corner=(0.7, 0.75), num_particles = 20000)
    mpm.add_analytic_box(min_corner=(0.4, 0.3), max_corner=(0.6, 0.5), rotation=3.1415926/4)
    mpm.add_analytic_box(min_corner=(0.0, 0.0), max_corner=(1.0, 0.05))


with Logger(directory + f'log.txt'):
    for frame in range(200):
        print("================== frame",frame,"===================")
        # mpm.step(8e-3)
        mpm.step(1/24)
        particles = mpm.particle_info()
        gui.circles(particles['position'],
                    radius=1.5,
                    color=0x068587)
        gui.show(directory + f'{frame:06d}.png')
        # print(mpm.getMaxVelocity())
        # gui.show()

ti.print_profile_info()