import os
import numpy as np
import taichi as ti

from projects.mpm.engine.mpm_solver_implicit import MPMSolverImplicit
from common.utils.logger import *

ti.init(arch=ti.cpu, default_fp=ti.f64)
# ti.init(arch=ti.gpu, default_fp=ti.f64)

test_case = 10
fps = 24
max_num_frame =200

directory = 'output/2d/' + str(test_case) + '/'
os.makedirs(directory, exist_ok=True)
gui = ti.GUI("PNMPM-2D", res=512, background_color=0x112F41)

mpm = MPMSolverImplicit(res=(128, 128))

if test_case == 0:  # jello drop on ground
    mpm.symplectic = True
    mpm.setDXandDT(DX=0.0078125, DT=0.00015625)
    mpm.setGravity((0, -1.0))
    mpm.setLameParameter(E=40, nu=0.2)

    mpm.add_cube(min_corner=(0.5, 0.6),
                 max_corner=(0.6, 0.9),
                 num_particles=3333,
                 rho=2)
    mpm.add_cube(min_corner=(0.4, 0.2),
                 max_corner=(0.7, 0.5),
                 num_particles=10000,
                 rho=2)
    mpm.add_analytic_box(min_corner=(0.0, 0.0), max_corner=(1.0, 0.05))

if test_case == 10:  # jello drop on mound
    mpm.symplectic = False
    mpm.cfl = 0.6
    mpm.setDXandDT(DX=0.00502513, DT=0.004)
    mpm.setGravity((0, -2.0))
    mpm.setLameParameter(E=40, nu=0.2)

    mpm.add_cube(min_corner=(0.3, 0.7),
                 max_corner=(0.7, 0.9),
                 num_particles=20000,
                 rho=2)
    mpm.add_analytic_box(min_corner=(0.4, 0.3),
                         max_corner=(0.6, 0.5),
                         rotation=(3.1415926 / 4, 0.0, 0.0, 0.0))
    mpm.add_analytic_box(min_corner=(0.0, 0.0), max_corner=(1.0, 0.05))

if test_case == 1:  # physbam snow
    mpm.symplectic = False
    mpm.cfl = 0.6
    mpm.setDXandDT(DX=0.00502513, DT=0.004)
    # mpm.symplectic = True
    # mpm.setDXandDT(DX=0.00502513, DT=0.0005)
    mpm.setGravity((0, -2.0))
    mpm.setLameParameter(E=40, nu=0.2)

    mpm.load_state(filename='data/2d.txt')
    mpm.add_analytic_box(min_corner=(0.4, 0.3),
                         max_corner=(0.6, 0.5),
                         rotation=(3.1415926 / 4, 0.0, 0.0, 0.0))
    mpm.add_analytic_box(min_corner=(0.0, 0.0), max_corner=(1.0, 0.05))


with Logger(directory + f'log.txt'):
    for frame in range(max_num_frame):
        print("================== frame", frame, "===================")
        # mpm.step(8e-3)
        mpm.step(1 / fps)
        particles = mpm.particle_info()
        gui.circles(particles['position'], radius=1.5, color=0x068587)
        gui.show(directory + f'{frame:06d}.png')
        # print(mpm.getMaxVelocity())
        # gui.show()

ti.print_profile_info()
