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
import scipy.optimize
from scipy.spatial import Delaunay

class TrajectoryOpt:

    def __init__(self, directory):
        test_case = 1007
        settings = read(test_case)
        settings['directory'] = directory
        self.simulator = StaticShell(settings=settings)
        self.design_variable = self.shrink_y(self.simulator.X.to_numpy().flatten())
        data = scipy.io.loadmat('input/wing_tips.mat')
        self.target_inner_trajectory = data['inner']
        self.target_outer_trajectory = data['outer']
        self.four_vertices = settings['four_vertices'] # [c, v1, v2, v3, v4] The last index is the end of the valley fold

        self.inner_vertex = settings['inner_vertex']
        self.outer_vertex = settings['outer_vertex']

        self.polygons = settings['polygons']
        self.polygon_triangles = settings['polygon_triangles']

        self.inner_trajectory = []
        self.outer_trajectory = []

        self.bar_length = np.linalg.norm(self.target_outer_trajectory[0] - self.target_inner_trajectory[0])
        cp_length = np.linalg.norm(self.design_variable[2 * self.inner_vertex: 2 * self.inner_vertex + 2] - self.design_variable[2 * self.outer_vertex: 2 * self.outer_vertex + 2])
        self.design_variable *= (self.bar_length / cp_length)
        self.target_inner_pos = []
        self.target_outer_pos = []

        self.n_segments = 100
        self.simulator.output_X(0)
    
    def triangulation(self, x):
        x_stack = x.reshape((self.simulator.n_particles, 2))
        elements = self.simulator.vertices.to_numpy()
        for i, poly in enumerate(self.polygons):
            points = x_stack[poly]
            tri = Delaunay(points)
            for j, tri_local in enumerate(tri.simplices):
                elements[self.polygon_triangles[i][j]] = np.array([poly[tri_local[2]], poly[tri_local[1]], poly[tri_local[0]]])
        self.simulator.vertices.from_numpy(elements)
        self.simulator.reset()
    
    def shrink_y(self, x_vec):
        ''' x_vec is a 1D vector of (xxx, 3*n)'''
        return x_vec.reshape([self.simulator.n_particles, 3])[:, [0, 2]].flatten()

    def extend_y(self, x_vec):
        ''' x_vec is a 1D vector of (xxx, 2*n)'''
        result = np.zeros([self.simulator.n_particles, 3])
        result[:, [0, 2]] = x_vec.reshape([self.simulator.n_particles, 2])
        return result.flatten()

    def project_direction(self, x, direction):
        x_stack = x.reshape((self.simulator.n_particles, 2))
        direction_stack = direction.reshape((self.simulator.n_particles, 2))
        new_direction = direction_stack.copy()
        for i in range(len(self.four_vertices)):
            v0 = x_stack[self.four_vertices[i, 0]] + direction_stack[self.four_vertices[i, 0]] # moved position
            v1 = x_stack[self.four_vertices[i, 1]] + direction_stack[self.four_vertices[i, 1]]
            v2 = x_stack[self.four_vertices[i, 2]] + direction_stack[self.four_vertices[i, 2]]
            v3 = x_stack[self.four_vertices[i, 3]] + direction_stack[self.four_vertices[i, 3]]
            v4 = x_stack[self.four_vertices[i, 4]]
            d1 = (v1 - v0) / np.linalg.norm(v1 - v0)
            d2 = (v2 - v0) / np.linalg.norm(v2 - v0)
            d3 = (v3 - v0) / np.linalg.norm(v3 - v0)
            alpha1 = np.arccos(d1.dot(d2))
            alpha2 = np.arccos(d2.dot(d3))
            alpha4 = np.pi - alpha2
            d4_new = np.array([[np.cos(alpha4), -np.sin(alpha4)], [np.sin(alpha4), np.cos(alpha4)]]).dot(d1)
            v4_new = (v4 - v0).dot(d4_new) * d4_new + v0
            new_direction[self.four_vertices[i, 4]] = v4_new - v4
        return new_direction.flatten()

    
    def forward(self):
        print("[Opt] forward pass")
        X = self.extend_y(self.design_variable)
        self.simulator.X.from_numpy(X.reshape([self.simulator.n_particles, 3]))
        self.simulator.x.from_numpy(X.reshape([self.simulator.n_particles, 3]))
        self.simulator.reset()
        self.simulator.output_x(0)
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
        print("[Opt] update icp")
        # find touched indices and transformation
        T, indices1, indices2 = icp(np.array(self.inner_trajectory), np.array(self.outer_trajectory), self.target_inner_trajectory, self.target_outer_trajectory)
        # transform the target curves to match the source
        T = np.linalg.inv(T)
        registered_inner = (np.dot(T[:3, :3], self.target_inner_trajectory.T) + T[:3, 3:4]).T
        registered_outer = (np.dot(T[:3, :3], self.target_outer_trajectory.T) + T[:3, 3:4]).T
        self.target_inner_pos = registered_inner[indices1]
        # self.target_inner_pos[0] = registered_inner[0]
        # self.target_inner_pos[-1] = registered_inner[-1]
        self.target_outer_pos = registered_outer[indices1]
        # self.target_outer_pos[0] = registered_outer[0]
        # self.target_outer_pos[-1] = registered_outer[-1]

    def save_trajectory(self, f):
        scipy.io.savemat(self.simulator.directory + f'caches/opt_traj_{f:06d}.mat', mdict={'inner': self.inner_trajectory, 'outer': self.outer_trajectory, 'inner_target': self.target_inner_pos, 'outer_target': self.target_outer_pos})
    
    def save_state(self, f):
        scipy.io.savemat(self.simulator.directory + f'caches/opt_state_{f:06d}.mat', mdict={'design_variable': self.design_variable})
    
    def loss(self):
        print("[Opt] evaluate loss")
        trajectory_loss = 0.0
        for i in range(len(self.inner_trajectory)):
            diff = self.inner_trajectory[i] - self.target_inner_pos[i]
            trajectory_loss += 0.5 * diff.dot(diff)
            diff = self.outer_trajectory[i] - self.target_outer_pos[i]
            trajectory_loss += 0.5 * diff.dot(diff)
        return trajectory_loss
    
    def gradient(self):
        print("[Opt] evaluate gradient")
        grad = np.zeros([self.simulator.n_particles * 3])
        for i in range(len(self.inner_trajectory)): 
            dLidx = np.zeros([self.simulator.n_particles * 3])
            dLidx[self.inner_vertex * 3: self.inner_vertex * 3 + 3] = self.inner_trajectory[i] - self.target_inner_pos[i]
            dLidx[self.outer_vertex * 3: self.outer_vertex * 3 + 3] = self.outer_trajectory[i] - self.target_outer_pos[i]
            self.simulator.load_state(i)
            if i != 0:
                dLidX = self.simulator.compute_dLdX(dLidx)
                grad += dLidX
            else:
                grad += dLidx
        return self.shrink_y(grad)

    def constraint_fun(self, x):
        print("[Opt] evaluate constraint")
        x_vec = self.extend_y(x).reshape((self.simulator.n_particles, 3))
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
        print("[Opt] evaluate constraint jacobian")
        x_vec = self.extend_y(x).reshape((self.simulator.n_particles, 3))
        result = np.zeros((len(self.four_vertices), 2 * self.simulator.n_particles))
        for i in range(len(self.four_vertices)):
            v0 = x_vec[self.four_vertices[i, 0]]
            v1 = x_vec[self.four_vertices[i, 1]]
            v2 = x_vec[self.four_vertices[i, 2]]
            v3 = x_vec[self.four_vertices[i, 3]]
            v4 = x_vec[self.four_vertices[i, 4]]
            
            da0 = angle_gradient(v0, v1, v2)
            result[i, self.four_vertices[i, 0]*2: self.four_vertices[i, 0]*2+2] += da0[[0, 2]]
            result[i, self.four_vertices[i, 1]*2: self.four_vertices[i, 1]*2+2] += da0[[3, 5]]
            result[i, self.four_vertices[i, 2]*2: self.four_vertices[i, 2]*2+2] += da0[[6, 8]]
            
            da1 = angle_gradient(v0, v2, v3)
            result[i, self.four_vertices[i, 0]*2: self.four_vertices[i, 0]*2+2] -= da1[[0, 2]]
            result[i, self.four_vertices[i, 2]*2: self.four_vertices[i, 2]*2+2] -= da1[[3, 5]]
            result[i, self.four_vertices[i, 3]*2: self.four_vertices[i, 3]*2+2] -= da1[[6, 8]]

            da2 = angle_gradient(v0, v3, v4)
            result[i, self.four_vertices[i, 0]*2: self.four_vertices[i, 0]*2+2] += da2[[0, 2]]
            result[i, self.four_vertices[i, 3]*2: self.four_vertices[i, 3]*2+2] += da2[[3, 5]]
            result[i, self.four_vertices[i, 4]*2: self.four_vertices[i, 4]*2+2] += da2[[6, 8]]

            da3 = angle_gradient(v0, v4, v1)
            result[i, self.four_vertices[i, 0]*2: self.four_vertices[i, 0]*2+2] -= da3[[0, 2]]
            result[i, self.four_vertices[i, 4]*2: self.four_vertices[i, 4]*2+2] -= da3[[3, 5]]
            result[i, self.four_vertices[i, 1]*2: self.four_vertices[i, 1]*2+2] -= da3[[6, 8]]
        return result
    
    def optimize(self):
        # https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#tutorial-sqlsp
        eq_cons = {'type': 'eq',
                   'fun': self.constraint_fun,
                   'jac': self.constraint_jac}

        def func(x):
            self.design_variable[:] = x
            self.forward()
            return self.loss()
        def gradient(x):
            return self.gradient()

        x = self.design_variable.copy()
        # self.triangulation(x)
        self.simulator.output_X(0)
        self.forward()
        self.update_icp()
        self.save_state(0)
        self.save_trajectory(0)
        step_size = 1e-4
        for i in range(1000):
            # self.triangulation(x)
            energy = func(x)
            grad = gradient(x)
            direction = - step_size * grad
            new_direction = self.project_direction(x, direction)
            x += new_direction
            self.design_variable[:] = x
            self.simulator.X.from_numpy(self.extend_y(x).reshape([self.simulator.n_particles, 3]))
            self.simulator.output_X(i+1)
            self.save_state(i+1)
            self.save_trajectory(i+1)
            print(f"[Optimization] Loss: {energy:.6f}")

    # def invert_loss(self):

    # def invert_loss()

    def diff_test_objective(self):
        self.simulator.newton_tol = 1e-8
        self.n_segments = 10
        self.forward()
        self.update_icp()
        def energy(x):
            self.design_variable[:] = x
            self.forward()
            return self.loss()
        def gradient(x):
            self.design_variable[:] = x
            self.forward()
            return self.gradient()
        check_gradient(self.design_variable, energy, gradient, eps=1e-4)
    
    
    def diff_test_constraint(self):
        min_edge_length = 10000000
        X = self.simulator.X.to_numpy()
        for edge in self.simulator.edges.to_numpy():
            le = np.linalg.norm(X[edge[0]] - X[edge[1]])
            min_edge_length = min(le, min_edge_length)
        for sec in self.four_vertices:
            direction = np.random.random(3)
            direction[1] = 0
            direction = 0.5 * min_edge_length * direction / np.linalg.norm(direction)
            X[sec[0]] += direction
        check_jacobian(self.shrink_y(X.flatten()), self.constraint_fun, self.constraint_jac, 3)

 
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
        opt.optimize()
        # opt.diff_test_constraint()