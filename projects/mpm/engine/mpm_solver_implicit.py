import taichi as ti
import numpy as np
import time
from common.physics.fixed_corotated import *
from common.math.math_tools import *

import math

import scipy.sparse
import scipy.sparse.linalg

@ti.func
def linear_offset(offset):
    return (offset[0]+2)*5+(offset[1]+2)

@ti.data_oriented
class MPMSolverImplicit:

    grid_size = 4096

    def __init__(
            self,
            res,
            size=1):
        self.dim = len(res)

        self.res = res
        self.n_particles = ti.field(dtype=ti.i32, shape=())
        self.dx = size / res[0]
        self.inv_dx = 1.0/self.dx
        self.dt = 2e-2 * self.dx / size
        # self.dt = 0.001
        self.p_vol = self.dx**self.dim
        self.p_rho = 1000
        self.p_mass = self.p_vol * self.p_rho
        max_num_particles = 2**27
        self.gravity = ti.Vector.field(self.dim, dtype=ti.f64, shape=())
        self.pid = ti.var(ti.i32)

        # position
        self.x = ti.Vector.field(self.dim, dtype=ti.f64)
        # velocity
        self.v = ti.Vector.field(self.dim, dtype=ti.f64)
        # affine velocity field
        self.C = ti.Matrix.field(self.dim, self.dim, dtype=ti.f64)
        # deformation gradient
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=ti.f64)

        self.Jp = ti.field(dtype=ti.f64)

        if self.dim == 2:
            indices = ti.ij
        else:
            indices = ti.ijk

        offset = tuple(-self.grid_size // 2 for _ in range(self.dim))
        self.offset = offset


        # grid node momentum/velocity
        self.grid_v = ti.Vector.field(self.dim, dtype=ti.f64)
        self.grid_f = ti.Vector.field(self.dim, dtype=ti.f64)
        self.grid_idx = ti.field(dtype=ti.i32)
        # grid node mass
        self.grid_m = ti.field(dtype=ti.f64)

        self.num_active_grid = ti.field(dtype=ti.i32, shape=())

        grid_block_size = 128
        self.grid = ti.root.pointer(indices, self.grid_size // grid_block_size) # 32

        if self.dim == 2:
            self.leaf_block_size = 16
        else:
            self.leaf_block_size = 8

        block = self.grid.pointer(indices,
                                    grid_block_size // self.leaf_block_size) # 8

        def block_component(c):
            block.dense(indices, self.leaf_block_size).place(c, offset=offset) # 16 (-2048, 2048)

        block_component(self.grid_m)
        for v in self.grid_v.entries:
            block_component(v)
        for f in self.grid_f.entries:
            block_component(f)
        block_component(self.grid_idx)
            

        block.dynamic(ti.indices(self.dim),
                      1024 * 1024,
                      chunk_size=self.leaf_block_size**self.dim * 8).place(
                          self.pid, offset=offset + (0, ))

        self.particle = ti.root.dynamic(ti.i, max_num_particles, 2**20)
        self.particle.place(self.x, self.v, self.C, self.F, self.Jp)

        # Sparse Matrix for Newton iteration
        MAX_LINEAR = 5000000
        # self.data_rhs = ti.field(ti.f64, shape=n_particles * dim)
        self.data_row = ti.field(ti.i32, shape=MAX_LINEAR)
        self.data_col = ti.field(ti.i32, shape=MAX_LINEAR)
        self.data_val = ti.field(ti.f64, shape=MAX_LINEAR)
        self.data_x = ti.field(ti.f64, shape=MAX_LINEAR)
        self.entryCol = ti.field(ti.i32, shape=MAX_LINEAR)
        self.entryVal = ti.Matrix.field(self.dim, self.dim, ti.f64, shape=MAX_LINEAR)
        self.isbound = ti.field(ti.i32, shape=MAX_LINEAR)

        self.dof2idx = ti.Vector.field(self.dim, ti.i32, shape=100000000)
        self.num_entry = ti.field(ti.i32, shape=())
        


        # Young's modulus and Poisson's ratio
        self.E, self.nu = 1e5 * size, 0.2
        # Lame parameters
        self.mu_0, self.lambda_0 = self.E / (
            2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) *
                                                    (1 - 2 * self.nu))


        # variables in newton optimization
        self.total_E = ti.field(ti.f64, shape=())
        self.dv = ti.field(ti.f64, shape=MAX_LINEAR)
        self.ddv = ti.field(ti.f64, shape=MAX_LINEAR)
        self.rhs = ti.field(ti.f64, shape=MAX_LINEAR)


        self.analytic_collision = []
        self.grid_collidable_objects = []
        

        self.gravity[None][1] = -20.0



    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    @ti.kernel
    def build_pid(self):
        ti.block_dim(64)
        for p in self.x:
            base = int(ti.floor(self.x[p] * self.inv_dx - 0.5))
            ti.append(self.pid.parent(), base - ti.Vector(list(self.offset)),
                      p)

    @ti.kernel
    def p2g(self, dt: ti.f64):     
        ti.no_activate(self.particle)
        ti.block_dim(256)
        ti.cache_shared(*self.grid_v.entries)
        ti.cache_shared(self.grid_m)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

            # snow projection
            U, sig, V = ti.svd(self.F[p])
            for d in ti.static(range(self.dim)):
                new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                  1 + 4.5e-3)  # Plasticity
                self.Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
            self.F[p] = U @ sig @ V.T()

            # Loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.grid_v[base + offset] += weight * self.p_mass * self.v[p]
                self.grid_m[base + offset] += weight * self.p_mass

    @ti.kernel
    def grid_normalization_and_gravity(self, dt: ti.f64):
        self.num_active_grid[None] = 0

        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:  # No need for epsilon here
                self.grid_v[I] = (1 / self.grid_m[I]
                                  ) * self.grid_v[I]  # Momentum to velocity
                idx = self.num_active_grid[None].atomic_add(1) # Avoid error in parallel computing
                # print(idx)
                self.grid_idx[I] = idx
                self.dof2idx[idx] = I
        

    def grid_dof(self):
        nn = 0
        for i in range(100):
            for j in range(100):
                if self.grid_m[i,j]>0:
                    # print(i,j,nn)
                    self.grid_idx[i,j] = nn
                    self.dof2idx[nn] = [i,j]
                    nn = nn + 1

        if not nn == self.num_active_grid[None]:
            print("ERROR: Inconsistent num of active grid")



    @ti.kernel
    def explicit_force(self, dt: ti.f64):
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            dw = [fx-1.5, -2*(fx-1), fx-0.5]

            # Hardening coefficient: snow gets harder when compressed
            h = ti.exp(10 * (1.0 - self.Jp[p]))
            mu, la = self.mu_0 * h, self.lambda_0 * h

            # mu, la = self.mu_0, self.lambda_0
            # U, sig, V = ti.svd(self.F[p])
            # J = self.F[p].determinant()
            # P = 2 * mu * (self.F[p] - U @ V.T()) @ self.F[p].T(
            #     ) + ti.Matrix.identity(ti.f64, self.dim) * la * J * (J - 1)
            P = elasticity_first_piola_kirchoff_stress(self.F[p], la, mu)
            P = P @ self.F[p].transpose()

            for offset in ti.static(ti.grouped(self.stencil_range())):
                weight = 1.0
                dN = ti.Vector.zero(ti.f64, self.dim)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                if ti.static(self.dim == 2):
                    dN[0] = dw[offset[0]][0]*w[offset[1]][1] * self.inv_dx
                    dN[1] = w[offset[0]][0]*dw[offset[1]][1] * self.inv_dx
                else:
                    dN[0] = dw[offset[0]][0]*w[offset[1]][1]*w[offset[2]][2] * self.inv_dx
                    dN[1] = w[offset[0]][0]*dw[offset[1]][1]*w[offset[2]][2] * self.inv_dx
                    dN[2] = w[offset[0]][0]*w[offset[1]][1]*dw[offset[2]][2] * self.inv_dx


                self.grid_f[base + offset] += - self.p_vol * P @ dN

    @ti.kernel
    def implicit_force(self, dt: ti.f64):
        # Build Matrix: Inertial term: dim*N in total
        for i in ti.ndrange(self.num_active_grid[None]):
            for j in range(25):
                self.entryCol[i*25+j] = -1
                self.entryVal[i*25+j] = ti.Matrix.zero(ti.f64, self.dim, self.dim)
            gid = self.dof2idx[i]
            m = self.grid_m[gid[0], gid[1]]
            self.entryCol[i*25+12] = i
            self.entryVal[i*25+12] = ti.Matrix.identity(ti.f64, self.dim) * m


        # Build Matrix: Loop over all particles
        for I in ti.grouped(self.pid):
            p = self.pid[I]

            # F = self.F[p]
            F = ti.Matrix.identity(ti.f64, self.dim)
            mu, la = self.mu_0, self.lambda_0
            dPdF = elasticity_first_piola_kirchoff_stress_derivative(F, la, mu) #* self.dt * self.dt * self.p_vol
            
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            dw = [fx-1.5, -2*(fx-1), fx-0.5]

            cnt = 0
            cached_w = []
            cached_node = []
            cached_idx = []
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dN = ti.Vector.zero(ti.f64, self.dim)
                dN[0] = dw[offset[0]][0]*w[offset[1]][1] * self.inv_dx
                dN[1] = w[offset[0]][0]*dw[offset[1]][1] * self.inv_dx
                cached_w.append(F.transpose() @ dN)
                cached_idx.append(self.grid_idx[base + offset])
                cached_node.append(offset)
                cnt += 1
            for i in ti.static(range(9)):
                wi = cached_w[i]
                dofi = cached_idx[i]
                nodei = cached_node[i]
                for j in ti.static(range(9)):
                    wj = cached_w[j]
                    dofj = cached_idx[j]
                    nodej = cached_node[j]

                    dFdX = ti.Matrix.zero(ti.f64, self.dim, self.dim)
                    for q in ti.static(range(self.dim)):
                        for v in ti.static(range(self.dim)):
                            dFdX[0,0] += dPdF[self.dim*v+0,self.dim*q+0]*wi[v]*wj[q]
                            dFdX[0,1] += dPdF[self.dim*v+0,self.dim*q+1]*wi[v]*wj[q]
                            dFdX[1,0] += dPdF[self.dim*v+1,self.dim*q+0]*wi[v]*wj[q]
                            dFdX[1,1] += dPdF[self.dim*v+1,self.dim*q+1]*wi[v]*wj[q]
                    dFdX = dFdX * self.p_vol * self.dt * self.dt
                    ioffset = dofi*25+linear_offset(nodei-nodej)
                    joffset = dofj*25+linear_offset(nodej-nodei)
                    if dofi < dofj:                    
                        self.entryCol[ioffset] = dofj
                        self.entryVal[ioffset] += dFdX   
                        self.entryCol[joffset] = dofi
                        self.entryVal[joffset] += dFdX.transpose()     
                    elif dofi == dofj:
                        self.entryCol[ioffset] = dofj
                        self.entryVal[ioffset] += dFdX                        


    @ti.kernel
    def implicit_update(self, dt:ti.f64):
        # for i in range(self.num_active_grid[None]):
        #     dv = ti.Vector([self.data_x[i*2], self.data_x[i*2+1]])
        #     gid = self.dof2idx[i]
        #     self.grid_v[gid[0], gid[1]] += dv     

        for i in range(self.num_active_grid[None]):
            dv = ti.Vector([self.dv[i*2], self.dv[i*2+1]])
            gid = self.dof2idx[i]
            self.grid_v[gid[0], gid[1]] += dv    

    @ti.kernel
    def explicit_update(self, dt:ti.f64):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:  # No need for epsilon here                
                self.grid_v[I] += dt * self.gravity[None]
                self.grid_v[I] += dt * self.grid_f[I] * (1 / self.grid_m[I])

    @ti.kernel
    def build_T(self):
        ndof = self.num_active_grid[None]
        d = self.dim
        for i in range(ndof):
            for k in range(25):
                c = i*25+k
                j = self.entryCol[i*25+k]
                M = self.entryVal[i*25+k]
                if not j == -1:
                    self.data_row[c*4] = i*2
                    self.data_col[c*4] = j*2
                    self.data_val[c*4] = M[0,0]

                    self.data_row[c*4+1] = i*2
                    self.data_col[c*4+1] = j*2+1
                    self.data_val[c*4+1] = M[0,1]

                    self.data_row[c*4+2] = i*2+1
                    self.data_col[c*4+2] = j*2
                    self.data_val[c*4+2] = M[1,0]

                    self.data_row[c*4+3] = i*2+1
                    self.data_col[c*4+3] = j*2+1
                    self.data_val[c*4+3] = M[1,1]
                                 
    def solve_system(self, dt):
        ndof = self.num_active_grid[None]
        d = self.dim
        rhs = np.zeros(ndof*d)
        for i in range(ndof):
            f = self.grid_f[self.dof2idx[i][0],self.dof2idx[i][1]]
            rhs[i*d], rhs[i*d+1] = f[0] * self.dt, f[1] * self.dt
            m = self.grid_m[self.dof2idx[i][0], self.dof2idx[i][1]]
            rhs[i*d+1] += (-10.0)*self.dt*m

        nentry = ndof * d * 25 * 4
        # print("aaaaaaaaaaaaaaaaaaaaa", ndof, np.max(self.data_row.to_numpy()), np.max(self.data_col.to_numpy()))

        # print(self.data_row.to_numpy()[0:nentry], self.data_col.to_numpy()[0:nentry])

        A = scipy.sparse.csr_matrix((self.data_val.to_numpy()[0:nentry], (self.data_row.to_numpy()[0:nentry], self.data_col.to_numpy()[0:nentry])), shape=(ndof*d,ndof*d))
        
        # A = scipy.sparse.lil_matrix((ndof*d, ndof*d))
        # print("a", np.max(self.data_col.to_numpy()[0:75]), np.max(self.data_row.to_numpy()[0:75]))
        # print(self.data_col[25+12])
        # for i in range(ndof*d):
        #     A[i,i] = 1.0
        # print("ndof", ndof)
        # print(self.data_col[4*12])
        # for i in range(ndof):
        #     print(A[i,i])
        # for i in range(ndof):
        #     gid = self.dof2idx[i]
        #     m = self.grid_m[gid[0], gid[1]]
        #     A[i*d, i*d] = m
        #     A[i*d+1, i*d+1] = m
        data_x = scipy.sparse.linalg.spsolve(A, rhs)


        for i in range(ndof):
            self.data_x[i*d], self.data_x[i*d+1] = data_x[i*d], data_x[i*d+1]


    @ti.kernel
    def total_energy(self, dt:ti.f64):
        '''
            Compute total energy of MPM system
        '''
        pass

    @ti.kernel
    def BuildMatrix(self, dt:ti.f64):
        # Build Matrix: Inertial term: dim*N in total
        for i in ti.ndrange(self.num_active_grid[None]):
            for j in range(25):
                self.entryCol[i*25+j] = -1
                self.entryVal[i*25+j] = ti.Matrix.zero(ti.f64, self.dim, self.dim)
            gid = self.dof2idx[i]
            m = self.grid_m[gid[0], gid[1]]
            self.entryCol[i*25+12] = i
            self.entryVal[i*25+12] = ti.Matrix.identity(ti.f64, self.dim) * m


        # Build Matrix: Loop over all particles
        for I in ti.grouped(self.pid):
            p = self.pid[I]

            F = self.F[p]
            mu, la = self.mu_0, self.lambda_0
            dPdF = elasticity_first_piola_kirchoff_stress_derivative(F, la, mu) #* self.dt * self.dt * self.p_vol

            
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            dw = [fx-1.5, -2*(fx-1), fx-0.5]

            cnt = 0
            cached_w = []
            cached_node = []
            cached_idx = []
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dN = ti.Vector.zero(ti.f64, self.dim)
                dN[0] = dw[offset[0]][0]*w[offset[1]][1] * self.inv_dx
                dN[1] = w[offset[0]][0]*dw[offset[1]][1] * self.inv_dx
                cached_w.append(F.transpose() @ dN)
                cached_idx.append(self.grid_idx[base + offset])
                cached_node.append(offset)
                cnt += 1

            for i in ti.static(range(9)):
                wi = cached_w[i]
                dofi = cached_idx[i]
                nodei = cached_node[i]
                for j in ti.static(range(9)):
                    wj = cached_w[j]
                    dofj = cached_idx[j]
                    nodej = cached_node[j]


                    dFdX = ti.Matrix.zero(ti.f64, self.dim, self.dim)
                    for q in ti.static(range(self.dim)):
                        for v in ti.static(range(self.dim)):
                            dFdX[0,0] += dPdF[self.dim*v+0,self.dim*q+0]*wi[v]*wj[q]
                            dFdX[0,1] += dPdF[self.dim*v+0,self.dim*q+1]*wi[v]*wj[q]
                            dFdX[1,0] += dPdF[self.dim*v+1,self.dim*q+0]*wi[v]*wj[q]
                            dFdX[1,1] += dPdF[self.dim*v+1,self.dim*q+1]*wi[v]*wj[q]
                    dFdX = dFdX * self.p_vol * self.dt * self.dt


                    ioffset = dofi*25+linear_offset(nodei-nodej)
                    self.entryCol[ioffset] = dofj
                    self.entryVal[ioffset] += dFdX

                    # ioffset = dofi*25+linear_offset(nodei-nodej)
                    # joffset = dofj*25+linear_offset(nodej-nodei)

                    # if dofi < dofj:                    
                    #     self.entryCol[ioffset] = dofj
                    #     self.entryVal[ioffset] += dFdX   
                    #     self.entryCol[joffset] = dofi
                    #     self.entryVal[joffset] += dFdX.transpose() 
                    # elif dofi == dofj:
                    #     self.entryCol[ioffset] = dofj
                    #     self.entryVal[ioffset] += dFdX

        ndof = self.num_active_grid[None]
        padding = 3
        for i in range(ndof):
            # print(self.rhs[i*self.dim], self.rhs[i*self.dim+1])
            gid = self.dof2idx[i]
            if gid[0] < padding or gid[1] < padding:
                srt = i*25
                end = srt + 25
                for k in range(srt, end):
                    j = self.entryCol[k]
                    if i == j:
                        self.entryVal[k] = ti.Matrix.identity(ti.f64, self.dim)
                    elif not j == -1:
                        self.entryCol[k] = -1
                        self.entryVal[k] = ti.Matrix.zero(ti.f64, self.dim, self.dim)  
                        srt2 = j*25
                        end2 = srt2+25
                        for k2 in range(srt2, end2):
                            i2 = self.entryCol[k2]
                            if i2 == i:
                                self.entryCol[k2] = -1
                                self.entryVal[k2] = ti.Matrix.zero(ti.f64, self.dim, self.dim)  

    @ti.kernel
    def ComputeResidual(self, dt:ti.f64):
        ndof = self.num_active_grid[None]
        d = self.dim
        for i in ti.ndrange(ndof):
            gid = self.dof2idx[i]
            m = self.grid_m[gid[0], gid[1]]
            f = self.grid_f[gid[0], gid[1]]

            self.rhs[i*d+0] = self.dv[i*d+0] * m - self.dt * f[0]
            self.rhs[i*d+1] = self.dv[i*d+1] * m - self.dt * f[1] - self.dt * m * (-10.0)            

        ndof = self.num_active_grid[None]
        padding = 3
        for i in range(ndof):
            # print(self.rhs[i*self.dim], self.rhs[i*self.dim+1])
            gid = self.dof2idx[i]
            for d in ti.static(range(self.dim)):
                if gid[d] < padding:
                    self.rhs[i*self.dim+d] = 0

    @ti.kernel
    def ProjectBC(self):
        ndof = self.num_active_grid[None]
        padding = 3
        for i in range(ndof):
            # print(self.rhs[i*self.dim], self.rhs[i*self.dim+1])
            gid = self.dof2idx[i]
            for d in ti.static(range(self.dim)):
                if gid[d] < padding and self.rhs[i*self.dim+d] > 0:
                    self.rhs[i*self.dim+d] = 0
                    # # print("aaa", self.rhs[i*self.dim+d])
                    # idx = i*self.dim+d
                    # for s in range(ndof*d*25*4):
                    #     if (self.data_col[s] == idx or self.data_row[s] == idx) and not (self.data_col[s] == self.data_row[s]):
                    #         self.data_col[s] = 0
                    #         self.data_row[s] = 0
                    #         self.data_val[s] = 0
        # padding = 3
        # for I in ti.grouped(self.grid_m):
        #     for d in ti.static(range(self.dim)):
        #         if False:
        #             if I[d] < -self.grid_size // 2 + padding and self.grid_v[
        #                     I][d] < 0:
        #                 self.grid_v[I][d] = 0  # Boundary conditions
        #             if I[d] >= self.grid_size // 2 - padding and self.grid_v[
        #                     I][d] > 0:
        #                 self.grid_v[I][d] = 0
        #         else:
        #             if I[d] < padding and self.grid_v[I][d] < 0:
        #                 self.grid_v[I][d] = 0  # Boundary conditions
        #             if I[d] >= self.res[d] - padding and self.grid_v[I][
        #                     d] > 0:
        #                 self.grid_v[I][d] = 0     

    @ti.kernel
    def TotalEnergy(self):
        '''
            Compute total elastic energy
        '''
        e = 0.0
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            F = self.F[p]
            la, mu = self.lambda_0, self.mu_0
            psi = elasticity_energy(F, la, mu)
            e += psi
        print(e)

    def SolveLinearSystem(self, dt):
        ndof = self.num_active_grid[None]
        d = self.dim
        rhs = np.zeros(ndof*d)
        rhs = self.rhs.to_numpy()[0:ndof*d]

        nentry = ndof * d * 25 * 4
        A = scipy.sparse.csr_matrix((self.data_val.to_numpy()[0:nentry], (self.data_row.to_numpy()[0:nentry], self.data_col.to_numpy()[0:nentry])), shape=(ndof*d,ndof*d))
        x = scipy.sparse.linalg.spsolve(A, -rhs)


        for i in range(ndof):
            self.data_x[i*d], self.data_x[i*d+1] = x[i*d], x[i*d+1]        

    @ti.kernel
    def UpdateState(self, dt:ti.f64):
        pass

        # ndof = self.num_active_grid[None]
        # d = self.dim
        # rhs = np.zeros(ndof*d)
        # for i in range(ndof):
        #     f = self.grid_f[self.dof2idx[i][0],self.dof2idx[i][1]]
        #     rhs[i*d], rhs[i*d+1] = f[0] * self.dt, f[1] * self.dt
        #     m = self.grid_m[self.dof2idx[i][0], self.dof2idx[i][1]]
        #     rhs[i*d+1] += (-10.0)*self.dt*m

    @ti.kernel
    def LineSearch(self, dt:ti.f64):
        # self.dv += self.data_x
        for i in range(self.num_active_grid[None] * self.dim):
            self.dv[i] += self.data_x[i] * 0.1

    @ti.kernel
    def BuildInitial(self, dt:ti.f64):
        '''
            Set initial guess for newton iteration
        '''
        # ndof = self.num_active_grid[None]
        # padding = 3
        # for i in range(ndof):
        #     gid = self.dof2idx[i]
        #     old_v = self.grid_v[gid]
        #     vi = ti.Vector([old_v[0], old_v[1]])
        #     if gid[0] < padding or gid[1] < padding:
        #         vi[1] = 0.0
        #         vi = vi - old_v
        #         self.dv[i*2] = vi[0]
        #         self.dv[i*2+1] = vi[1] 
        #     else:
        #         self.dv[i*2]=0
        #         self.dv[i*2+1]=(-10.0)*self.dt

        ndof = self.num_active_grid[None]
        for i in range(ndof):
            gid = self.dof2idx[i]
            x = gid * self.dx
            old_v = self.grid_v[gid]
            flag, vi = self.analytic_collision[0](x, old_v)
            # flag = 0
            # vi = ti.Vector([old_v[0], old_v[1]])
            # for k in ti.static(range(len(self.analytic_collision))):
            #     f, vn = self.analytic_collision[k](x, vi)
            #     if f == 1:
            #         flag = 1
            #         vi = vn
            if flag == 1:
                vi = vi - old_v
                self.dv[i*2] = vi[0]
                self.dv[i*2+1] = vi[1]
            else:
                g = self.gravity[None]
                self.dv[i*2] = g[0]*self.dt
                self.dv[i*2+1] = g[1]*self.dt



    def implicit_newton(self, dt):
        # perform one full newton
        self.ddv.fill(0)
        self.rhs.fill(0)
        self.dv.fill(0)
        self.BuildInitial(self.dt)
        # Newton iteration
        for iter in range(1):
            print('iter = ', iter)
            self.ComputeResidual(dt) # Compute g
            self.BuildMatrix(dt) # Compute H
            self.data_col.fill(0)
            self.data_row.fill(0)
            self.build_T()

            self.SolveLinearSystem(dt)  # solve dx = H^(-1)g

            self.LineSearch(dt)
        
        self.implicit_update(dt)
        # self.TotalEnergy()

    @ti.kernel
    def grid_collision(self, dt: ti.f64):
        padding = 3
        for I in ti.grouped(self.grid_m):
            for d in ti.static(range(self.dim)):
                if False:
                    if I[d] < -self.grid_size // 2 + padding and self.grid_v[
                            I][d] < 0:
                        self.grid_v[I][d] = 0  # Boundary conditions
                    if I[d] >= self.grid_size // 2 - padding and self.grid_v[
                            I][d] > 0:
                        self.grid_v[I][d] = 0
                else:
                    if I[d] < padding and self.grid_v[I][d] < 0:
                        self.grid_v[I][d] = 0  # Boundary conditions
                    if I[d] >= self.res[d] - padding and self.grid_v[I][
                            d] > 0:
                        self.grid_v[I][d] = 0

    @ti.kernel
    def collide(self):
        for i in ti.static(range(len(self.analytic_collision))):
            vnew = self.analytic_collision[i](ti.Vector([0.0,1.0]))
            # print(vnew)


    @ti.kernel
    def g2p(self, dt: ti.f64):
        ti.block_dim(256)
        ti.cache_shared(*self.grid_v.entries)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2
            ]
            dw = [fx-1.5, -2*(fx-1), fx-0.5]
            new_v = ti.Vector.zero(ti.f64, self.dim)
            new_C = ti.Matrix.zero(ti.f64, self.dim, self.dim)
            new_F = ti.Matrix.zero(ti.f64, self.dim, self.dim)
            # loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(float) - fx
                g_v = self.grid_v[base + offset]
                weight = 1.0
                dN = ti.Vector.zero(ti.f64, self.dim)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                if ti.static(self.dim == 2):
                    dN[0] = dw[offset[0]][0]*w[offset[1]][1] * self.inv_dx
                    dN[1] = w[offset[0]][0]*dw[offset[1]][1] * self.inv_dx
                else:
                    dN[0] = dw[offset[0]][0]*w[offset[1]][1]*w[offset[2]][2] * self.inv_dx
                    dN[1] = w[offset[0]][0]*dw[offset[1]][1]*w[offset[2]][2] * self.inv_dx
                    dN[2] = w[offset[0]][0]*w[offset[1]][1]*dw[offset[2]][2] * self.inv_dx

                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
                new_F += g_v.outer_product(dN)
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += dt * self.v[p]  # advection
            self.F[p] *= (ti.Matrix.identity(ti.f64, self.dim) + self.dt * new_F)



    def advanceOneStepExplicit(self, dt):
        self.grid.deactivate_all()
        self.build_pid()
        self.p2g(dt)
        self.grid_normalization_and_gravity(dt)
        self.explicit_force(dt)
        self.explicit_update(dt)
        # self.grid_collision(dt)
        for p in self.grid_collidable_objects:
            p(dt)
        self.g2p(dt)

    def advanceOneStepNewton(self, dt):
        self.grid.deactivate_all()
        self.build_pid()
        self.p2g(dt)
        self.grid_normalization_and_gravity(dt)
        self.explicit_force(dt) 

        self.implicit_newton(dt)

        self.g2p(dt) 

    def step(self, frame_dt, print_stat=False):
        begin_t = time.time()
        substeps = int(frame_dt / self.dt) + 1
        # print("dt=", self.dt)
        for i in range(substeps):
            dt = frame_dt / substeps

            # self.advanceOneStepExplicit(dt)        

            self.advanceOneStepNewton(dt)

    ########### Sample MPM Particle Cloud ###########

    @ti.func
    def seed_particle(self, i, pos, velocity):
        self.x[i] = pos
        self.v[i] = velocity
        self.F[i] = ti.Matrix.identity(ti.f64, self.dim)

        self.Jp[i] = 1

    def add_cube(self, pos, size):
        a = ti.Vector(pos)
        b = ti.Vector(size)

        self.sample_cube(a, b)

    @ti.kernel
    def sample_cube(self, lower_corner: ti.template(), cube_size: ti.template()):
        new_particles = 20000
        for i in range(self.n_particles[None], self.n_particles[None] + new_particles):
            x = ti.Vector.zero(ti.f64, self.dim)
            for k in ti.static(range(self.dim)):
                x[k] = lower_corner[k] + ti.random()*cube_size[k]
            self.seed_particle(i, x, ti.Vector.zero(ti.f64, self.dim))
        self.n_particles[None] = self.n_particles[None] + new_particles


    @ti.kernel
    def copy_dynamic_nd(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in self.x:
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]

    def particle_info(self):
        np_x = np.ndarray((self.n_particles[None], self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_x, self.x)
        np_v = np.ndarray((self.n_particles[None], self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_v, self.v)

        return {
            'position': np_x,
            'velocity': np_v,
        }


    # Analytic Collision Objects
  
    def add_surface_collider(self,
                             point,
                             normal):
        point = list(point)
        # normalize normal
        normal_scale = 1.0 / math.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)


        @ti.func
        def get_velocity(x, v): # Compute new velocity
            flag = 0
            offset = x - ti.Vector(point)
            n = ti.Vector(normal)
            if offset.dot(n) < 0:
                flag = 1
                v = ti.Vector.zero(ti.f64, self.dim)
            return flag, v
            # normal_component = n.dot(v)
            # v = v - n * normal_component
            # return v
            # if x < padding

        self.analytic_collision.append(get_velocity)

        @ti.kernel
        def collide(dt: ti.f64):
            for I in ti.grouped(self.grid_m):
                offset = I * self.dx - ti.Vector(point)
                n = ti.Vector(normal)
                if offset.dot(n) < 0:
                    self.grid_v[I] = ti.Vector.zero(ti.f64, self.dim)

        self.grid_collidable_objects.append(collide) 


    @ti.kernel
    def test(self):
        print(self.f(ti.Vector([0.35,0.34]),ti.Vector([0.35,0.34])))

    def add_analytic_box(self):

        min_corner = ti.Vector([0.4, 0.3])
        max_corner = ti.Vector([0.6, 0.5])

        b = (min_corner + max_corner) / 2
        half_edge = (max_corner - min_corner) / 2

        theta = 3.1415926/4.0
        R = ti.Matrix([[ti.cos(theta), -ti.sin(theta)],[ti.sin(theta), ti.cos(theta)]])



        @ti.func
        def signedDistance(x):
            xx = R.transpose() @ (x - b)
            d = ti.Vector.zero(ti.f64, self.dim)
            dd = -100000.0
            for p in ti.static(range(self.dim)):
                d[p] = ti.abs(xx[p]) - half_edge[p]
                if dd < d[p]:
                    dd = d[p]
                d[p] = ti.max(d[p], 0.0)
            if dd > 0.0:
                dd = 0.0
            return dd + d.norm()



        @ti.func
        def particleCollision(x, v):
            # x_minus_b = x - b
            # X = R.transpose() @ x_minus_b
            flag = 0
            if signedDistance(x) < 0:
                flag = 1
                v = ti.Vector.zero(ti.f64, self.dim)
            return flag, v


        @ti.kernel
        def gridCollision(dt: ti.f64):
            for I in ti.grouped(self.grid_m):
                x = I * self.dx
                if signedDistance(x) < 0:
                    self.grid_v[I] = ti.Vector.zero(ti.f64, self.dim)
            
        # self.f = particleCollision
        self.analytic_collision.append(particleCollision)
        self.grid_collidable_objects.append(gridCollision)

