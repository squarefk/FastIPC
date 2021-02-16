from reader import *
import sys, os, time, math
from StaticShell import StaticShell
from logger import *
from timer import *
import taichi as ti
import scipy.io
from angle import angle, angle_gradient
from diff_test import check_gradient, check_jacobian, finite_gradient
from trajectory_icp import icp

class TrajectoryOpt:

    def __init__(self, directory):
        test_case = 1007
        settings = read(test_case)
        settings['directory'] = directory
        self.simulator = StaticShell(settings=settings)
        self.design_variable = self.simulator.X.to_numpy().flatten()
        data = scipy.io.loadmat('input/wing_tips.mat')
        self.target_inner_trajectory = data['inner']
        self.target_outer_trajectory = data['outer']
        self.four_vertices = settings['four_vertices']

        self.inner_vertex = settings['inner_vertex']
        self.outer_vertex = settings['outer_vertex']

        self.inner_trajectory = []
        self.outer_trajectory = []

        self.bar_length = np.linalg.norm(self.target_outer_trajectory[0] - self.target_inner_trajectory[0])
        cp_length = np.linalg.norm(self.design_variable[3 * self.inner_vertex: 3 * self.inner_vertex + 3] - self.design_variable[3 * self.outer_vertex: 3 * self.outer_vertex + 3])
        self.design_variable *= (self.bar_length / cp_length)
        self.target_inner_pos = []
        self.target_outer_pos = []

        self.n_segments = 100
    
    def forward(self):
        self.simulator.X.from_numpy(self.design_variable.reshape([self.simulator.n_particles, 3]))
        self.simulator.x.from_numpy(self.design_variable.reshape([self.simulator.n_particles, 3]))
        self.simulator.reset()
        self.simulator.output_x(0)
        self.simulator.output_X(0)
        
        self.inner_trajectory = [[self.simulator.x[self.inner_vertex][0], self.simulator.x[self.inner_vertex][1], self.simulator.x[self.inner_vertex][2]]]
        self.outer_trajectory = [[self.simulator.x[self.outer_vertex][0], self.simulator.x[self.outer_vertex][1], self.simulator.x[self.outer_vertex][2]]]
        self.simulator.save_state(0)
        for i in range(1, self.n_segments + 1):
            self.simulator.set_target_angle(i / self.n_segments)
            self.simulator.advance()
            self.simulator.save_state(i)
            self.simulator.output_x(i)
            self.inner_trajectory.append([self.simulator.x[self.inner_vertex][0], self.simulator.x[self.inner_vertex][1], self.simulator.x[self.inner_vertex][2]])
            self.outer_trajectory.append([self.simulator.x[self.outer_vertex][0], self.simulator.x[self.outer_vertex][1], self.simulator.x[self.outer_vertex][2]])
            Timer_Print()
        self.inner_trajectory = np.array(self.inner_trajectory)
        self.outer_trajectory = np.array(self.outer_trajectory)
    
    def update_icp(self):
        # find touched indices and transformation
        T, indices1, indices2 = icp(np.array(self.inner_trajectory), np.array(self.outer_trajectory), self.target_inner_trajectory, self.target_outer_trajectory)
        # transform the target curves to match the source
        T = np.linalg.inv(T)
        registered_inner = (np.dot(T[:3, :3], self.target_inner_trajectory.T) + T[:3, 3:4]).T
        registered_outer = (np.dot(T[:3, :3], self.target_outer_trajectory.T) + T[:3, 3:4]).T
        self.target_inner_pos = registered_inner[indices1]
        self.target_inner_pos[0] = registered_inner[0]
        self.target_inner_pos[-1] = registered_inner[-1]
        self.target_outer_pos = registered_outer[indices1]
        self.target_outer_pos[0] = registered_outer[0]
        self.target_outer_pos[-1] = registered_outer[-1]

    def save_trajectory(self, filename):
        scipy.io.savemat(filename, mdict={'inner': self.inner_trajectory, 'outer': self.outer_trajectory})

    def loss(self, x):
        trajectory_loss = 0.0
        for i in range(len(self.inner_trajectory)):
            diff = self.inner_trajectory[i] - self.target_inner_pos[i]
            trajectory_loss += 0.5 * diff.dot(diff)
            diff = self.outer_trajectory[i] - self.target_outer_pos[i]
            trajectory_loss += 0.5 * diff.dot(diff)
        return trajectory_loss
    
    def gradient(self, x):
        grad = np.zeros_like(x)
        for i in range(len(self.inner_trajectory)): 
            dLidx = np.zeros_like(x)
            dLidx[self.inner_vertex * 3: self.inner_vertex * 3 + 3] = self.inner_trajectory[i] - self.target_inner_pos[i]
            dLidx[self.outer_vertex * 3: self.outer_vertex * 3 + 3] = self.outer_trajectory[i] - self.target_outer_pos[i]
            self.simulator.load_state(i)
            if i != 0:
                dLidX = self.simulator.compute_dLdX(dLidx)
                grad += dLidX
            else:
                grad += dLidx
        return grad

    def constraint_fun(self, x):
        x_vec = x.reshape((self.simulator.n_particles, 3))
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
        x_vec = x.reshape((self.simulator.n_particles, 3))
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
    

    def diff_test(self):
        self.simulator.newton_tol = 1e-8
        self.n_segments = 10
        self.forward()
        self.update_icp()
        def energy(x):
            self.design_variable[:] = x
            self.forward()
            return self.loss(x)
        def gradient(x):
            self.design_variable[:] = x
            self.forward()
            return self.gradient(x)
        check_gradient(self.design_variable, energy, gradient, eps=1e-4)

            

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
        opt.diff_test()