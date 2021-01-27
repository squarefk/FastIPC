from reader import *
from common.math.math_tools import *
from timer import *
from logger import *

import sys, os, time, math
import taichi as ti
import numpy as np
import meshio
import pickle
import scipy.sparse
import scipy.sparse.linalg
from sksparse.cholmod import *
from dihedral_angle import *
from simplex_volume import *

real = ti.f64

@ti.data_oriented
class StaticShell:
    codim = 2
    dim = 3
    X = ti.Vector.field(dim, real) # material space
    x = ti.Vector.field(dim, real) # world space
    vol = ti.field(real)
    rho = ti.field(real)
    la, mu = ti.field(real), ti.field(real)
    B = ti.Matrix.field(codim, codim, real)
    vertices = ti.field(ti.i32)
    edges = ti.field(ti.i32)
    rest_angle = ti.field(real)
    rest_e = ti.field(real)
    rest_h = ti.field(real)
    weight = ti.field(real)
    element_hessian = ti.field(real)
    element_indMap = ti.field(ti.int32)
    edge_hessian = ti.field(real)
    edge_indMap = ti.field(ti.int32)

    thickness = 0.0003
    base_bending_weight = 10
    E = 1e9
    nu = 0.3
    density = 800.

    xPrev = ti.Vector.field(dim, real) # to store start point in optimization

    def __init__(self, n_particles=None, n_elements=None, n_edges=None, **kwargs):
        
        if "settings" in kwargs:
            settings = kwargs.get("settings")
            n_particles = len(settings['mesh_particles'])
            n_elements = len(settings['mesh_elements'])
            n_edges = len(settings['mesh_edges'])

        self.n_particles = n_particles
        self.n_elements = n_elements
        self.n_edges = n_edges

        ti.root.dense(ti.i, n_particles).place(self.x, self.X, self.xPrev)
        ti.root.dense(ti.i, n_elements).place(self.la, self.mu, self.vol, self.rho, self.B)
        ti.root.dense(ti.ij, (n_elements, self.dim)).place(self.vertices)
        ti.root.dense(ti.ij, (n_edges, 5)).place(self.edges)
        ti.root.dense(ti.i, n_edges).place(self.rest_angle, self.rest_e, self.rest_h, self.weight)
        ti.root.dense(ti.ijk, (n_elements, 9, 9)).place(self.element_hessian)
        ti.root.dense(ti.ijk, (n_edges, 12, 12)).place(self.edge_hessian)
        ti.root.dense(ti.ij, (n_elements, 9)).place(self.element_indMap)
        ti.root.dense(ti.ij, (n_edges, 12)).place(self.edge_indMap)

        MAX_LINEAR = 144 * n_edges + 81 * n_elements + 10
        self.data_rhs = ti.field(real, shape=n_particles * self.dim)
        self.data_row = ti.field(ti.i32, shape=MAX_LINEAR)
        self.data_col = ti.field(ti.i32, shape=MAX_LINEAR)
        self.data_val = ti.field(real, shape=MAX_LINEAR)
        self.cnt = ti.field(ti.i32, shape=())

        self.hessian_selected_row = ti.field(ti.i32, shape=MAX_LINEAR)
        self.hessian_selected_col = ti.field(ti.i32, shape=MAX_LINEAR)
        self.hessian_selected_val = ti.field(real, shape=MAX_LINEAR)

        self.data_sol = ti.field(real, shape=n_particles * self.dim)
        self.dfx = ti.field(ti.i32, shape=n_particles * self.dim)

        self.mu.from_numpy(np.full(n_elements, self.E / (2 * (1 + self.nu))))
        self.la.from_numpy(np.full(n_elements, self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))))
        self.rho.from_numpy(np.full(n_elements, self.density))

        self.directory = ""
        
        if "settings" in kwargs:
            settings = kwargs.get("settings")
            self.X.from_numpy(settings['mesh_particles'].astype(np.float64))
            self.x.from_numpy(settings['mesh_particles'].astype(np.float64))
            self.vertices.from_numpy(settings['mesh_elements'].astype(np.int32))
            self.edges.from_numpy(settings['mesh_edges'].astype(np.int32))
            D = np.stack((settings['dirichlet'],) * self.dim, axis=-1).reshape((n_particles * self.dim))
            self.dfx.from_numpy(D.astype(np.int32))
            self.directory = settings['directory']
        
        self.set_target_angle(0.0)
        self.set_bending_weight(10.0)


    @ti.kernel
    def reset(self):
        for i in range(self.n_elements):
            ab = self.X[self.vertices[i, 1]] - self.X[self.vertices[i, 0]]
            ac = self.X[self.vertices[i, 2]] - self.X[self.vertices[i, 0]]
            T = ti.Matrix.cols([ab, ac])
            self.B[i] = (T.transpose() @ T).inverse()
            self.vol[i] = self.thickness * (ab.cross(ac)).norm() / 2
        
        for i in range(self.n_edges):
            self.rest_e[i] = (self.X[self.edges[i, 0]] - self.X[self.edges[i, 1]]).norm()
            if self.edges[i, 3] < 0:
                self.rest_h[i] = 1
                continue
            self.rest_angle[i] = 0.0
            X0 = self.X[self.edges[i, 2]]
            X1 = self.X[self.edges[i, 0]]
            X2 = self.X[self.edges[i, 1]]
            X3 = self.X[self.edges[i, 3]]
            n1 = (X1 - X0).cross(X2 - X0)
            n2 = (X2 - X3).cross(X1 - X3)
            self.rest_h[i] = (n1.norm() + n2.norm()) / (self.rest_e[i] * 6)                

    @ti.func
    def compute_T(self, i):
        ab = self.x[self.vertices[i, 1]] - self.x[self.vertices[i, 0]]
        ac = self.x[self.vertices[i, 2]] - self.x[self.vertices[i, 0]]
        T = ti.Matrix.cols([ab, ac])
        return T.transpose() @ T

    @ti.kernel
    def check_edge_error(self) -> real:
        edge_error = 0
        for e in range(self.n_edges):
            l = (self.x[self.edges[e, 0]] - self.x[self.edges[e, 1]]).norm()
            edge_error += ti.abs(l - self.rest_e[e]) / self.rest_e[e]
        return edge_error
    
    @ti.kernel
    def set_target_angle(self, percentage: real):
        for e in range(self.n_edges):
            self.rest_angle[e] = self.edges[e, 4] * np.pi * percentage
    
    @ti.kernel
    def set_bending_weight(self, base_bending_weight: real):
        for i in range(self.n_edges):
            if self.edges[i, 4] == 1:
                self.weight[i] = 100 * base_bending_weight
            elif self.edges[i, 4] == -1:
                self.weight[i] = 10 * base_bending_weight
            else:
                self.weight[i] = base_bending_weight
    
    @ti.kernel
    def compute_energy(self) -> real:
        total_energy = 0.0
        
        # membrane
        for e in range(self.n_elements):
            F = self.compute_T(e) @ self.B[e]
            lnJ = 0.5 * ti.log(F.determinant())
            mem = 0.5 * mu[e] * (F.trace() - 2 - 2 * lnJ) + 0.5 * la[e] * lnJ * lnJ
            total_energy += mem * self.vol[e]
        
        # bending
        for e in range(self.n_edges):
            if self.edges[e, 3] < 0: continue
            x0 = self.x[self.edges[e, 2]]
            x1 = self.x[self.edges[e, 0]]
            x2 = self.x[self.edges[e, 1]]
            x3 = self.x[self.edges[e, 3]]
            theta = dihedral_angle(x0, x1, x2, x3, self.edges[e, 4])
            ben = ((theta - self.rest_angle[e]) ** 2) * self.rest_e[e] / self.rest_h[e]
            total_energy += self.weight[e] * ben
        
        return total_energy
    
    @ti.kernel
    def compute_gradient_impl(self):
        # membrane
        for e in range(self.n_elements):
            x1, x2, x3 = self.x[self.vertices[e, 0]], self.x[self.vertices[e, 1]], self.x[self.vertices[e, 2]]
            A = self.compute_T(e)
            IA = A.inverse()
            IB = self.B[e]
            lnJ = 0.5 * ti.log(A.determinant() * IB.determinant())
            de_div_dA = ti.Vector([0.0, 0.0, 0.0, 0.0])
            for i in ti.static(range(2)):
                for j in ti.static(range(2)):
                    de_div_dA[j * self.codim + i] = self.vol[e] * ((0.5 * self.mu[e] * IB[i,j] + 0.5 * (-self.mu[e] + self.la[e] * lnJ) * IA[i,j]))
            Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            dA_div_dx = ti.Matrix.rows([Z, Z, Z, Z])
            for i in ti.static(range(3)):
                dA_div_dx[0, 3 + i] += 2.0 * (x2[i] - x1[i])
                dA_div_dx[0, 0 + i] -= 2.0 * (x2[i] - x1[i])
                dA_div_dx[1, 6 + i] += (x2[i] - x1[i])
                dA_div_dx[1, 3 + i] += (x3[i] - x1[i])
                dA_div_dx[1, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
                dA_div_dx[2, 6 + i] += (x2[i] - x1[i])
                dA_div_dx[2, 3 + i] += (x3[i] - x1[i])
                dA_div_dx[2, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
                dA_div_dx[3, 6 + i] += 2.0 * (x3[i] - x1[i])
                dA_div_dx[3, 0 + i] -= 2.0 * (x3[i] - x1[i])
            grad = dA_div_dx.transpose() @ de_div_dA
            
            indMap = ti.Vector([self.vertices[e, 0] * 3, self.vertices[e, 0] * 3 + 1, self.vertices[e, 0] * 3 + 2,
                                self.vertices[e, 1] * 3, self.vertices[e, 1] * 3 + 1, self.vertices[e, 1] * 3 + 2,
                                self.vertices[e, 2] * 3, self.vertices[e, 2] * 3 + 1, self.vertices[e, 2] * 3 + 2])
            for i in ti.static(range(9)):
                if self.dfx[indMap[i]]:
                    grad[i] = 0
                self.data_rhs[indMap[i]] += grad[i]
        
        # bending
        for e in range(self.n_edges):
            if self.edges[e, 3] < 0: continue
            x0 = self.x[self.edges[e, 2]]
            x1 = self.x[self.edges[e, 0]]
            x2 = self.x[self.edges[e, 1]]
            x3 = self.x[self.edges[e, 3]]
            theta = dihedral_angle(x0, x1, x2, x3, self.edges[e, 4])
            grad = dihedral_angle_gradient(x0, x1, x2, x3)
            grad *= self.weight[e] * 2 * (theta - self.rest_angle[e]) * self.rest_e[e] / self.rest_h[e]
            
            indMap = ti.Vector([self.edges[e, 2] * 3, self.edges[e, 2] * 3 + 1, self.edges[e, 2] * 3 + 2,
                                self.edges[e, 0] * 3, self.edges[e, 0] * 3 + 1, self.edges[e, 0] * 3 + 2,
                                self.edges[e, 1] * 3, self.edges[e, 1] * 3 + 1, self.edges[e, 1] * 3 + 2,
                                self.edges[e, 3] * 3, self.edges[e, 3] * 3 + 1, self.edges[e, 3] * 3 + 2])
            for i in ti.static(range(12)):
                if self.dfx[indMap[i]]:
                    grad[i] = 0
                self.data_rhs[indMap[i]] += grad[i]
    
    @ti.kernel
    def compute_hessian_impl(self, pd: ti.int32):
        self.cnt[None] = 0
        # membrane
        Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ahess = [ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z]), 
                ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z]), 
                ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z]), 
                ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z])]
        for d in ti.static(range(self.dim)):
            ahess[0][0 + d, 0 + d] += 2.0
            ahess[0][3 + d, 3 + d] += 2.0
            ahess[0][0 + d, 3 + d] -= 2.0
            ahess[0][3 + d, 0 + d] -= 2.0

            ahess[1][3 + d, 6 + d] += 1.0
            ahess[1][6 + d, 3 + d] += 1.0
            ahess[1][0 + d, 3 + d] -= 1.0
            ahess[1][0 + d, 6 + d] -= 1.0
            ahess[1][3 + d, 0 + d] -= 1.0
            ahess[1][6 + d, 0 + d] -= 1.0
            ahess[1][0 + d, 0 + d] += 2.0

            ahess[2][3 + d, 6 + d] += 1.0
            ahess[2][6 + d, 3 + d] += 1.0
            ahess[2][0 + d, 3 + d] -= 1.0
            ahess[2][0 + d, 6 + d] -= 1.0
            ahess[2][3 + d, 0 + d] -= 1.0
            ahess[2][6 + d, 0 + d] -= 1.0
            ahess[2][0 + d, 0 + d] += 2.0

            ahess[3][0 + d, 0 + d] += 2.0
            ahess[3][6 + d, 6 + d] += 2.0
            ahess[3][0 + d, 6 + d] -= 2.0
            ahess[3][6 + d, 0 + d] -= 2.0

        for e in range(self.n_elements):
            x1, x2, x3 = self.x[self.vertices[e, 0]], self.x[self.vertices[e, 1]], self.x[self.vertices[e, 2]]
            IB = self.B[e]
            A = self.compute_T(e)
            Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            dA_div_dx = ti.Matrix.rows([Z, Z, Z, Z])
            for i in ti.static(range(3)):
                dA_div_dx[0, 3 + i] += 2.0 * (x2[i] - x1[i])
                dA_div_dx[0, 0 + i] -= 2.0 * (x2[i] - x1[i])
                dA_div_dx[1, 6 + i] += (x2[i] - x1[i])
                dA_div_dx[1, 3 + i] += (x3[i] - x1[i])
                dA_div_dx[1, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
                dA_div_dx[2, 6 + i] += (x2[i] - x1[i])
                dA_div_dx[2, 3 + i] += (x3[i] - x1[i])
                dA_div_dx[2, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
                dA_div_dx[3, 6 + i] += 2.0 * (x3[i] - x1[i])
                dA_div_dx[3, 0 + i] -= 2.0 * (x3[i] - x1[i])
            
            IA = A.inverse()
            ainvda = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for endI in ti.static(range(3)):
                for dimI in ti.static(range(3)):
                    ainvda[endI * self.dim + dimI] = \
                        dA_div_dx[0, endI * self.dim + dimI] * IA[0, 0] + \
                        dA_div_dx[1, endI * self.dim + dimI] * IA[1, 0] + \
                        dA_div_dx[2, endI * self.dim + dimI] * IA[0, 1] + \
                        dA_div_dx[3, endI * self.dim + dimI] * IA[1, 1]
            deta = A.determinant()
            lnJ = 0.5 * ti.log(deta * IB.determinant())
            term1 = (-self.mu[e] + self.la[e] * lnJ) * 0.5
            hessian = (-term1 + 0.25 * self.la[e]) * (ainvda @ ainvda.transpose())
            aderivadj = ti.Matrix.rows([Z, Z, Z, Z])
            for d in ti.static(range(9)):
                aderivadj[0, d] = dA_div_dx[3, d]
                aderivadj[1, d] = - dA_div_dx[1, d]
                aderivadj[2, d] = - dA_div_dx[2, d]
                aderivadj[3, d] = dA_div_dx[0, d]
            hessian += term1 / deta * aderivadj.transpose() @ dA_div_dx
            for i in ti.static(range(2)):
                for j in ti.static(range(2)):
                    hessian += (term1 * IA[i, j] + 0.5 * self.mu[e] * IB[i, j]) * ahess[i + j * 2]
            hessian *= self.vol[e]
        
            if pd == 1:
                hessian = project_pd(hessian)
            
            indMap = ti.Vector([self.vertices[e, 0] * 3, self.vertices[e, 0] * 3 + 1, self.vertices[e, 0] * 3 + 2,
                                self.vertices[e, 1] * 3, self.vertices[e, 1] * 3 + 1, self.vertices[e, 1] * 3 + 2,
                                self.vertices[e, 2] * 3, self.vertices[e, 2] * 3 + 1, self.vertices[e, 2] * 3 + 2])
            
            for i in ti.static(range(9)):
                for j in ti.static(range(9)):
                    c = self.cnt[None] + e * 81 + i * 9 + j
                    val = hessian[i, j]
                    if self.dfx[indMap[i]] or self.dfx[indMap[j]]:
                        if i == j:
                            val = 1
                        else:
                            val = 0
                    self.data_row[c], self.data_col[c], self.data_val[c] = indMap[i], indMap[j], val
            
            for i in ti.static(range(9)):
                for j in ti.static(range(9)):        
                    if self.dfx[indMap[i]]:  # select row
                        c = self.cnt[None] + e * 81 + i * 9 + j
                        selected_val = hessian[i, j]
                        if self.dfx[indMap[j]]:
                            if i == j:
                                selected_val = 1
                            else:
                                selected_val = 0
                        self.hessian_selected_row[c], self.hessian_selected_col[c], self.hessian_selected_val[c] = indMap[i], indMap[j], selected_val

        self.cnt[None] += 81 * self.n_elements

        # bending
        for e in range(self.n_edges):
            if self.edges[e, 3] < 0: continue
            x0 = self.x[self.edges[e, 2]]
            x1 = self.x[self.edges[e, 0]]
            x2 = self.x[self.edges[e, 1]]
            x3 = self.x[self.edges[e, 3]]
            theta = dihedral_angle(x0, x1, x2, x3, self.edges[e, 4])
            grad = dihedral_angle_gradient(x0, x1, x2, x3)
            hessian = dihedral_angle_hessian(x0, x1, x2, x3)
            hessian *= self.weight[e] * 2.0 * (theta - self.rest_angle[e]) * self.rest_e[e] / self.rest_h[e]
            hessian += (self.weight[e] * 2.0 * self.rest_e[e] / self.rest_h[e]) * grad @ grad.transpose()
            
            if pd: 
                hessian = project_pd(hessian)
            indMap = ti.Vector([self.edges[e, 2] * 3, self.edges[e, 2] * 3 + 1, self.edges[e, 2] * 3 + 2,
                                self.edges[e, 0] * 3, self.edges[e, 0] * 3 + 1, self.edges[e, 0] * 3 + 2,
                                self.edges[e, 1] * 3, self.edges[e, 1] * 3 + 1, self.edges[e, 1] * 3 + 2,
                                self.edges[e, 3] * 3, self.edges[e, 3] * 3 + 1, self.edges[e, 3] * 3 + 2])
            for i in ti.static(range(12)):
                for j in ti.static(range(12)):
                    c = self.cnt[None] + e * 144 + i * 12 + j
                    val = hessian[i, j]
                    if self.dfx[indMap[i]] or self.dfx[indMap[j]]:
                        if i == j:
                            val = 1
                        else:
                            val = 0
                    self.data_row[c], self.data_col[c], self.data_val[c] = indMap[i], indMap[j], val
            
            for i in ti.static(range(12)):
                for j in ti.static(range(12)):
                    c = self.cnt[None] + e * 144 + i * 12 + j        
                    if self.dfx[indMap[i]]:  # select row
                        selected_val = hessian[i, j]
                        if self.dfx[indMap[j]]:
                            if i == j:
                                selected_val = 1
                            else:
                                selected_val = 0
                        self.hessian_selected_row[c], self.hessian_selected_col[c], self.hessian_selected_val[c] = indMap[i], indMap[j], selected_val

        self.cnt[None] += 144 * self.n_edges
    
    @ti.kernel
    def compute_hessian_Xx_impl(self):
        self.cnt[None] = 0
        # membrane
        for e in range(self.n_elements):
            x1, x2, x3 = self.x[self.vertices[e, 0]], self.x[self.vertices[e, 1]], self.x[self.vertices[e, 2]]
            X1, X2, X3 = self.X[self.vertices[e, 0]], self.X[self.vertices[e, 1]], self.X[self.vertices[e, 2]]
            IB = self.B[e]
            A = self.compute_T(e)
            IA = A.inverse()
            lnJ = 0.5 * ti.log(A.determinant() * IB.determinant())
            dv_div_dX = self.thickness * simplex_volume_gradient(X1, X2, X3)
            de_div_dA = ti.Vector([0.,0.,0.,0.])
            for i in ti.static(range(2)):
                for j in ti.static(range(2)):
                    de_div_dA[j * self.codim + i] = ((0.5 * self.mu[e] * IB[i,j] + 0.5 * (-self.mu[e] + self.la[e] * lnJ) * IA[i,j]))
            Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            dA_div_dx = ti.Matrix.rows([Z, Z, Z, Z])
            dB_div_dX = ti.Matrix.rows([Z, Z, Z, Z])
            for i in ti.static(range(3)):
                dA_div_dx[0, 3 + i] += 2.0 * (x2[i] - x1[i])
                dA_div_dx[0, 0 + i] -= 2.0 * (x2[i] - x1[i])
                dA_div_dx[1, 6 + i] += (x2[i] - x1[i])
                dA_div_dx[1, 3 + i] += (x3[i] - x1[i])
                dA_div_dx[1, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
                dA_div_dx[2, 6 + i] += (x2[i] - x1[i])
                dA_div_dx[2, 3 + i] += (x3[i] - x1[i])
                dA_div_dx[2, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
                dA_div_dx[3, 6 + i] += 2.0 * (x3[i] - x1[i])
                dA_div_dx[3, 0 + i] -= 2.0 * (x3[i] - x1[i])

                dB_div_dX[0, 3 + i] += 2.0 * (X2[i] - X1[i])
                dB_div_dX[0, 0 + i] -= 2.0 * (X2[i] - X1[i])
                dB_div_dX[1, 6 + i] += (X2[i] - X1[i])
                dB_div_dX[1, 3 + i] += (X3[i] - X1[i])
                dB_div_dX[1, 0 + i] += - (X2[i] - X1[i]) - (X3[i] - X1[i])
                dB_div_dX[2, 6 + i] += (X2[i] - X1[i])
                dB_div_dX[2, 3 + i] += (X3[i] - X1[i])
                dB_div_dX[2, 0 + i] += - (X2[i] - X1[i]) - (X3[i] - X1[i])
                dB_div_dX[3, 6 + i] += 2.0 * (X3[i] - X1[i])
                dB_div_dX[3, 0 + i] -= 2.0 * (X3[i] - X1[i])
                
            de_div_dx = dA_div_dx.transpose() @ de_div_dA

            # first term
            hessian = dv_div_dX @ de_div_dx.transpose()

            # second term
            Z4 = ti.Vector([0.,0.,0.,0.])
            dbinv_div_db = ti.Matrix.rows([Z4, Z4, Z4, Z4])
            for m in ti.static(range(2)):
                for n in ti.static(range(2)):
                    for i in ti.static(range(2)):
                        for j in ti.static(range(2)): 
                            dbinv_div_db[n * self.codim + m, j * self.codim + i] = - IB[m, i] * IB[j, n]
            
            d2e_divA_divB = 0.5 * self.mu[e] * dbinv_div_db
            for m in ti.static(range(2)):
                for n in ti.static(range(2)):
                    for i in ti.static(range(2)):
                        for j in ti.static(range(2)):
                            d2e_divA_divB[n * self.codim + m, j * self.codim + i] -= 0.25 * self.la[e] * IA[m, n] * IB[j, i]
            hessian += self.vol[e] * dB_div_dX.transpose() @ d2e_divA_divB.transpose() @ dA_div_dx
            
            indMap = ti.Vector([self.vertices[e, 0] * 3, self.vertices[e, 0] * 3 + 1, self.vertices[e, 0] * 3 + 2,
                                self.vertices[e, 1] * 3, self.vertices[e, 1] * 3 + 1, self.vertices[e, 1] * 3 + 2,
                                self.vertices[e, 2] * 3, self.vertices[e, 2] * 3 + 1, self.vertices[e, 2] * 3 + 2])
            for i in ti.static(range(9)):
                for j in ti.static(range(9)):
                    if self.dfx[indMap[j]]:
                        hessian[i, j] = 0
                    c = self.cnt[None] + e * 81 + i * 9 + j
                    self.data_row[c], self.data_col[c], self.data_val[c] = indMap[i], indMap[j], hessian[i, j]
        self.cnt[None] += 81 * self.n_elements

        # bending
        for e in range(self.n_edges):
            if self.edges[e, 3] < 0: continue
            x1 = self.x[self.edges[e, 2]]
            x2 = self.x[self.edges[e, 0]]
            x3 = self.x[self.edges[e, 1]]
            x4 = self.x[self.edges[e, 3]]

            X1 = self.X[self.edges[e, 2]]
            X2 = self.X[self.edges[e, 0]]
            X3 = self.X[self.edges[e, 1]]
            X4 = self.X[self.edges[e, 3]]

            theta = dihedral_angle(x1, x2, x3, x4, self.edges[e, 4])
            grad = dihedral_angle_gradient(x1, x2, x3, x4)

            dA1_div_dX = simplex_volume_gradient(X1, X2, X3)
            dA2_div_dX = simplex_volume_gradient(X2, X3, X4)

            dAsum_div_dX = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for d in ti.static(range(3)):
                dAsum_div_dX[self.dim * 0 + d] = dA1_div_dX[self.dim * 0 + d] 
                dAsum_div_dX[self.dim * 1 + d] = dA1_div_dX[self.dim * 1 + d] + dA2_div_dX[self.dim * 0 + d]
                dAsum_div_dX[self.dim * 2 + d] = dA1_div_dX[self.dim * 2 + d] + dA2_div_dX[self.dim * 1 + d]
                dAsum_div_dX[self.dim * 3 + d] =                                dA2_div_dX[self.dim * 2 + d]
            
            n = (X2 - X3).normalized()
            dl_div_dX = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for d in ti.static(range(3)):
                dl_div_dX[self.dim * 1 + d] = n[d]
                dl_div_dX[self.dim * 2 + d] = - n[d]

            hessian = self.weight[e] * 2 * (theta - self.rest_angle[e]) / self.rest_h[e] *  (2 * dl_div_dX - dAsum_div_dX / (3 * self.rest_h[e])) @ grad.transpose()
            
            indMap = ti.Vector([self.edges[e, 2] * 3, self.edges[e, 2] * 3 + 1, self.edges[e, 2] * 3 + 2,
                                self.edges[e, 0] * 3, self.edges[e, 0] * 3 + 1, self.edges[e, 0] * 3 + 2,
                                self.edges[e, 1] * 3, self.edges[e, 1] * 3 + 1, self.edges[e, 1] * 3 + 2,
                                self.edges[e, 3] * 3, self.edges[e, 3] * 3 + 1, self.edges[e, 3] * 3 + 2])
            
            # project
            for i in ti.static(range(12)):
                for j in ti.static(range(12)):
                    if self.dfx[indMap[j]]:
                        hessian[i, j] = 0
                    c = self.cnt[None] + e * 144 + i * 12 + j
                    self.data_row[c], self.data_col[c], self.data_val[c] = indMap[i], indMap[j], hessian[i, j]
        self.cnt[None] += 144 * self.n_edges
    
    def compute_gradient(self):
        self.data_rhs.fill(0)
        self.compute_gradient_impl()
        return self.data_rhs.to_numpy()

    def compute_hessian(self, pd=1):
        self.data_row.fill(0)
        self.data_col.fill(0)
        self.data_val.fill(0)
        self.hessian_selected_col.fill(0)
        self.hessian_selected_row.fill(0)
        self.hessian_selected_val.fill(0)
        self.compute_hessian_impl(pd)
        row, col, val = self.data_row.to_numpy()[:self.cnt[None]], self.data_col.to_numpy()[:self.cnt[None]], self.data_val.to_numpy()[:self.cnt[None]]
        return scipy.sparse.csr_matrix((val, (row, col)), shape=(self.dim * self.n_particles, self.dim * self.n_particles))
    
    def select_hessian_rows(self):
        row, col, val = self.hessian_selected_row.to_numpy()[:self.cnt[None]], self.hessian_selected_col.to_numpy()[:self.cnt[None]], self.hessian_selected_val.to_numpy()[:self.cnt[None]]
        return scipy.sparse.csr_matrix((val, (row, col)), shape=(self.dim * self.n_particles, self.dim * self.n_particles))
    
    def compute_hessian_Xx(self):
        self.data_row.fill(0)
        self.data_col.fill(0)
        self.data_val.fill(0)
        self.compute_hessian_Xx_impl()
        row, col, val = self.data_row.to_numpy()[:self.cnt[None]], self.data_col.to_numpy()[:self.cnt[None]], self.data_val.to_numpy()[:self.cnt[None]]
        return scipy.sparse.csr_matrix((val, (row, col)), shape=(self.dim * self.n_particles, self.dim * self.n_particles))
    
    @ti.kernel
    def residual() -> real:
        residual = 0.0
        for i in range(self.n_particles):
            for d in ti.static(range(self.dim)):
                residual = max(residual, ti.abs(self.data_sol[i * self.dim + d]))
        return residual
    
    @ti.kernel
    def save_xPrev():
        for i in range(self.n_particles):
            self.xPrev[i] = self.x[i]
        
    @ti.kernel
    def apply_sol(alpha : real):
        for i in range(self.n_particles):
            for d in ti.static(range(self.dim)):
                self.x(d)[i] = self.xPrev(d)[i] + data_sol[i * self.dim + d] * alpha
    
    def output_x(self, f):
        particle_pos = self.x.to_numpy()
        vertices_ = self.vertices.to_numpy()
        meshio.write_points_cells(
            self.directory + f'objs/world_{f:06d}.obj',
            particle_pos,
            [("triangle", self.vertices.to_numpy())]
        )

    def output_X(self, f):
        particle_pos = self.X.to_numpy()
        vertices_ = self.vertices.to_numpy()
        meshio.write_points_cells(
            self.directory + f'objs/material_{f:06d}.obj',
            particle_pos,
            [("triangle", self.vertices.to_numpy())]
        )

    def advance():
        with Timer("Advance"):
            newton_iter = 0
            while True:
                newton_iter += 1
                print("-------------------- Newton Iteration: ", newton_iter, " --------------------")
                with Timer("Build System"):
                    neg_grad = - self.compute_gradient()
                    A = self.compute_hessian(1)
                with Timer("Solve System"):
                    factor = cholesky(A)
                    sol = factor(rhs)
                    self.data_sol.from_numpy(sol)
                residual = self.residual()
                print("Search Direction Residual : ", residual)
                if newton_iter > 1 and residual < 1e-3:
                    break
                elif residual < 1e-6:
                    break
                with Timer("Line Search"):
                    E0 = self.compute_energy()
                    self.save_xPrev()
                    alpha = 1
                    self.apply_sol(alpha)
                    E = self.compute_energy()
                    while E > E0:
                        alpha *= 0.5
                        self.apply_sol(alpha)
                        E = self.compute_energy()
                    print("[Step size after line search: ", alpha, "]")
            print("Edge Error: ", check_edge_error())
            newton_iter_total += newton_iter
            print("Avg Newton iter: ", newton_iter_total / (f + 1))
    
    def compute_dLdX(self, dLdx):
        H = self.compute_hessian(0)
        factor = cholesky(H)
        sol = factor(dLdx)
        HXx = self.compute_hessian_Xx()
        dLdX = -HXx.dot(sol)
        selected_rows = self.select_hessian_rows()
        dLdy = selected_rows.dot(sol)
        dLdX += dLdy
        return dLdX


    
if __name__ == "__main__":
    from diff_test import *
    ti.init(arch=ti.cpu, default_fp=real) #, cpu_max_num_threads=1
    simulator = StaticShell(4, 2, 1)
    mesh_particles = np.array([[0, 0, 0], 
                               [0, 2, 0],
                              [-1, 1, 0],
                               [1, 1, 1]], dtype=np.float64)
    mesh_elements = np.array([[0, 1, 2],
                              [0, 3, 1]], dtype=np.int32)
    mesh_edges = np.array([[0, 1, 2, 3, 1]], dtype=np.int32)

    simulator.x.from_numpy(2 * mesh_particles)
    simulator.X.from_numpy(mesh_particles)
    simulator.edges.from_numpy(mesh_edges)
    simulator.vertices.from_numpy(mesh_elements)
    simulator.la.from_numpy(np.ones(simulator.n_elements, dtype=np.float64))
    simulator.mu.from_numpy(np.ones(simulator.n_elements, dtype=np.float64))
    simulator.weight.from_numpy(np.ones(simulator.n_edges, dtype=np.float64))
    simulator.reset()

    dfx = simulator.dfx.to_numpy()
    # dfx[0] = 1
    # dfx[1] = 1
    simulator.dfx.from_numpy(dfx)
    simulator.compute_dLdX(np.ones(12))

    with Logger(f'test.txt'):
        def optimization_gradient(x):
            x_view = x.view().reshape((simulator.n_particles, 3))
            simulator.x.from_numpy(x_view)
            return simulator.compute_gradient()

        def optimization_hessian(x):
            x_view = x.view().reshape((simulator.n_particles, 3))
            simulator.x.from_numpy(x_view)
            return simulator.compute_hessian(0)
        
        check_jacobian(simulator.x.to_numpy().flatten(), optimization_gradient, optimization_hessian, 3 * simulator.n_particles)

        def diffshell_gradient(x):
            x_view = x.view().reshape((simulator.n_particles, 3))
            simulator.X.from_numpy(x_view)
            simulator.reset()
            return simulator.compute_gradient()

        def diffshell_hessian(x):
            x_view = x.view().reshape((simulator.n_particles, 3))
            simulator.X.from_numpy(x_view)
            simulator.reset()
            return simulator.compute_hessian_Xx().transpose()

        check_jacobian(simulator.X.to_numpy().flatten(), diffshell_gradient, diffshell_hessian, 3 * simulator.n_particles)