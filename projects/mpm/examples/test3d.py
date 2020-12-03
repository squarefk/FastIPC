import os
import numpy as np
import taichi as ti

from projects.mpm.engine.mpm_solver_implicit import MPMSolverImplicit
from common.utils.logger import *

ti.init(arch=ti.cpu, default_fp=ti.f64)

directory = 'output/3d/'
os.makedirs(directory, exist_ok=True)


gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)
mpm = MPMSolverImplicit(res=(64, 64, 64), size=10)


mpm.symplectic = False
mpm.setGravity((0,-100,0))


# mpm.add_cube(min_corner=(2, 6, 3), max_corner=(3, 7, 6))
mpm.add_cube(min_corner=(2, 1.0, 3), max_corner=(3, 2.0, 6))

mpm.add_surface_collider(point=(0.0, 0.15, 0.0), normal=(0.0, 1.0, 0.0))
# mpm.add_analytic_box(min_corner=(0.0, 0.0, 0.0), max_corner=(1.0, 0.05, 1.0))

def writeObjFile(f, X):
    fo = open(directory+"frame_"+str(f)+".obj", "w")
    for i in range(X.shape[0]):
        fo.write("v "+str(X[i][0])+" "+str(X[i][1])+" "+str(X[i][2])+"\n")
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