from reader import *
import sys, os, time, math
from StaticShell import StaticShell
from logger import *
from timer import *
import taichi as ti
import scipy.io
from angle import angle, angle_gradient
from diff_test import check_gradient, check_jacobian

class TrajectoryOpt:

    def __init__(self, directory):
        test_case = 1007
        settings = read(test_case)
        settings['directory'] = directory
        self.simulator = StaticShell(settings=settings)
        self.design_variable = self.simulator.X.to_numpy().flatten()
        data = scipy.io.loadmat('input/wing_tips.mat')
        self.target_inner_trajectory = data['left']
        self.target_outer_trajectory = data['right']
        self.four_vertices = settings['four_vertices']

    
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

    def constraint_fun(self, x):
        x_vec = x.view().reshape((self.simulator.n_particles, 3))
        result = np.zeros(len(self.four_vertices))
        for i in range(len(self.four_vertices)):
            v0 = x_vec[self.four_vertices[i, 0]]
            v1 = x_vec[self.four_vertices[i, 1]]
            v2 = x_vec[self.four_vertices[i, 2]]
            v3 = x_vec[self.four_vertices[i, 3]]
            v4 = x_vec[self.four_vertices[i, 4]]
            result[i] = angle(v0, v1, v2) - angle(v0, v2, v3) + angle(v0, v3, v4) - angle(v0, v4, v1)
        return result
    
    def constraint_jac(self, x):
        x_vec = x.view().reshape((self.simulator.n_particles, 3))
        result = np.zeros((len(self.four_vertices), 3 * self.simulator.n_particles))
        for i in range(len(self.four_vertices)):
            v0 = x_vec[self.four_vertices[i, 0]]
            v1 = x_vec[self.four_vertices[i, 1]]
            v2 = x_vec[self.four_vertices[i, 2]]
            v3 = x_vec[self.four_vertices[i, 3]]
            v4 = x_vec[self.four_vertices[i, 4]]
            
            da0 = angle_gradient(v0, v1, v2)
            result[i, self.four_vertices[i, 0]*3: self.four_vertices[i, 0]*3+3] += da0[0:3]
            result[i, self.four_vertices[i, 1]*3: self.four_vertices[i, 1]*3+3] += da0[3:6]
            result[i, self.four_vertices[i, 2]*3: self.four_vertices[i, 2]*3+3] += da0[6:9]
            
            da1 = angle_gradient(v0, v2, v3)
            result[i, self.four_vertices[i, 0]*3: self.four_vertices[i, 0]*3+3] -= da1[0:3]
            result[i, self.four_vertices[i, 2]*3: self.four_vertices[i, 2]*3+3] -= da1[3:6]
            result[i, self.four_vertices[i, 3]*3: self.four_vertices[i, 3]*3+3] -= da1[6:9]

            da2 = angle_gradient(v0, v3, v4)
            result[i, self.four_vertices[i, 0]*3: self.four_vertices[i, 0]*3+3] += da2[0:3]
            result[i, self.four_vertices[i, 3]*3: self.four_vertices[i, 3]*3+3] += da2[3:6]
            result[i, self.four_vertices[i, 4]*3: self.four_vertices[i, 4]*3+3] += da2[6:9]

            da3 = angle_gradient(v0, v4, v1)
            result[i, self.four_vertices[i, 0]*3: self.four_vertices[i, 0]*3+3] -= da3[0:3]
            result[i, self.four_vertices[i, 4]*3: self.four_vertices[i, 4]*3+3] -= da3[3:6]
            result[i, self.four_vertices[i, 1]*3: self.four_vertices[i, 1]*3+3] -= da3[6:9]
        return result
    
    def optimize():
        # https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#tutorial-sqlsp
        eq_cons = {'type': 'eq',
                   'fun': self.constraint_fun,
                   'jac': self.constraint_jac}
            

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
        # opt.forward()
        # opt.simulator.output_X(0)
        check_jacobian(opt.design_variable, opt.constraint_fun, opt.constraint_jac, 3)
    