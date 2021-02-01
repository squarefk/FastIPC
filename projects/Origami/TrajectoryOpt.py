from reader import *
import sys, os, time, math
from StaticShell import StaticShell
from logger import *
from timer import *
import taichi as ti

class TrajectoryOpt:

    def __init__(self, directory):
        test_case = 1006
        settings = read(test_case)
        settings['directory'] = directory
        self.simulator = StaticShell(settings=settings)
        self.trajectory = None
        self.design_variable = self.simulator.X.to_numpy().flatten()
    
    def forward(self):
        self.simulator.X.from_numpy(self.design_variable.view().reshape([self.simulator.n_particles, 3]))
        self.simulator.reset()
        self.simulator.x.from_numpy(self.design_variable.view().reshape([self.simulator.n_particles, 3]))
        for i in range(1, 101):
            self.simulator.set_target_angle(i / 100)
            self.simulator.advance()
            self.simulator.output_x(i)
            Timer_Print()
    
    def loss(self, x):
        self.design_variable[:] = x
        self.forward()
    
    def dloss_dxn(self, i):
        pass

    def gradient(self, x):
        self.design_variable[:] = x # copy x into design variable
        pass
            

if __name__ == "__main__":

    real = ti.f64
    ti.init(arch=ti.cpu, default_fp=real)
    directory = 'output/' + "wing/"
    os.makedirs(directory + 'images/', exist_ok=True)
    os.makedirs(directory + 'caches/', exist_ok=True)
    os.makedirs(directory + 'objs/', exist_ok=True)
    print('output directory:', directory)

    with Logger(directory + f'log.txt'):
        opt = TrajectoryOpt(directory)
        opt.forward()
    