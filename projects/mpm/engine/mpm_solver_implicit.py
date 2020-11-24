import taichi as ti
import numpy as np
import time
from common.physics.fixed_corotated import *
from common.math.math_tools import *

from common.utils.timer import *

import math

import scipy.sparse
import scipy.sparse.linalg
from projects.mpm.engine.sparse_matrix import SparseMatrix, CGSolver
##############################################################################
real = ti.f64

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
        # self.dx = size / res[0]
        self.dx = 0.00502513
        self.inv_dx = 1.0/self.dx
        self.dt = 2e-2 * self.dx / size
        # self.dt = 0.001

        max_num_particles = 2**27
        self.gravity = ti.Vector.field(self.dim, dtype=real, shape=())
        self.pid = ti.field(ti.i32)

        # position
        self.p_x = ti.Vector.field(self.dim, dtype=real)
        # velocity
        self.p_v = ti.Vector.field(self.dim, dtype=real)
        # affine velocity field
        self.p_C = ti.Matrix.field(self.dim, self.dim, dtype=real)
        # deformation gradient
        self.p_F = ti.Matrix.field(self.dim, self.dim, dtype=real)
        self.p_Fp = ti.Matrix.field(self.dim, self.dim, dtype=real)
        self.p_F_backup = ti.Matrix.field(self.dim, self.dim, dtype=real) # F^n backup
        # determinant of plastic
        self.p_Jp = ti.field(dtype=real)
        # volume
        self.p_vol = ti.field(dtype=real)
        # density
        self.p_rho = ti.field(dtype=real)
        # mass
        self.p_mass = ti.field(dtype=real)

        if self.dim == 2:
            indices = ti.ij
        else:
            indices = ti.ijk

        offset = tuple(-self.grid_size // 2 for _ in range(self.dim))
        self.offset = offset


        # grid node momentum/velocity
        self.grid_v = ti.Vector.field(self.dim, dtype=real)
        self.grid_f = ti.Vector.field(self.dim, dtype=real)
        self.grid_idx = ti.field(dtype=ti.i32)
        self.grid_dv = ti.Vector.field(self.dim, dtype=real)
        # grid node mass
        self.grid_m = ti.field(dtype=real)

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
        for dv in self.grid_dv.entries:
            block_component(dv)
        for f in self.grid_f.entries:
            block_component(f)
        block_component(self.grid_idx)
            

        block.dynamic(ti.indices(self.dim),
                      1024 * 1024,
                      chunk_size=self.leaf_block_size**self.dim * 8).place(
                          self.pid, offset=offset + (0, ))

        self.particle = ti.root.dynamic(ti.i, max_num_particles, 2**20)
        self.particle.place(self.p_x, self.p_v, self.p_C, self.p_F, self.p_Fp,
                                self.p_Jp, self.p_vol, self.p_rho, self.p_mass,
                                self.p_F_backup)

        # Sparse Matrix for Newton iteration
        MAX_LINEAR = 5000000
        # self.data_rhs = ti.field(real, shape=n_particles * dim)
        self.data_row = ti.field(ti.i32, shape=MAX_LINEAR)
        self.data_col = ti.field(ti.i32, shape=MAX_LINEAR)
        self.data_val = ti.field(real, shape=MAX_LINEAR)
        self.data_x = ti.field(real, shape=MAX_LINEAR)
        self.entryCol = ti.field(ti.i32, shape=MAX_LINEAR)
        self.entryVal = ti.Matrix.field(self.dim, self.dim, real, shape=MAX_LINEAR)
        self.isbound = ti.field(ti.i32, shape=MAX_LINEAR)

        self.nodeCNTol = ti.field(real, shape=MAX_LINEAR)

        self.dof2idx = ti.Vector.field(self.dim, ti.i32, shape=100000000)
        self.num_entry = ti.field(ti.i32, shape=())
        
        self.total_step = ti.field(ti.i32, shape=())

        # Young's modulus and Poisson's ratio
        E, nu = 1e5 * size, 0.2
        self.setLameParameter(40, 0.2)
        # # Lame parameters
        # self.mu_0, self.lambda_0 = self.E / (
        #     2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) *
        #                                             (1 - 2 * self.nu))


        # variables in newton optimization
        self.total_E = ti.field(real, shape=())
        self.dv = ti.field(real, shape=MAX_LINEAR)
        self.ddv = ti.field(real, shape=MAX_LINEAR)
        self.DV = ti.field(real, shape=MAX_LINEAR)
        self.rhs = ti.field(real, shape=MAX_LINEAR)

        self.boundary = ti.field(ti.i32, shape=MAX_LINEAR)

        self.result = ti.field(real, shape=MAX_LINEAR) # for debug purpose only

        self.matrix = SparseMatrix()
        self.cgsolver = CGSolver()

        self.analytic_collision = []
        self.grid_collidable_objects = []
        

        # self.gravity[None][1] = -20.0
        self.setGravity((0, -0.2))



    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    @ti.kernel
    def build_pid(self):
        ti.block_dim(64)
        for p in self.p_x:
            base = int(ti.floor(self.p_x[p] * self.inv_dx - 0.5))
            ti.append(self.pid.parent(), base - ti.Vector(list(self.offset)),
                      p)

    @ti.kernel
    def p2g(self, dt: real):     
        ti.no_activate(self.particle)
        ti.block_dim(256)
        ti.cache_shared(*self.grid_v.entries)
        ti.cache_shared(*self.grid_dv.entries)
        ti.cache_shared(self.grid_m)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.p_x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            fx = self.p_x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

            # # snow projection
            # U, sig, V = ti.svd(self.p_F[p])
            # for d in ti.static(range(self.dim)):
            #     new_sig = min(max(sig[d, d], 1 - 0.015),
            #                       1 + 0.005)  # Plasticity
            #     self.p_Jp[p] *= sig[d, d] / new_sig
            #     # sig[d, d] = new_sig
            # self.p_F[p] = U @ sig @ V.T()

            # plastic = True
            # if plastic:
            #     U, sig, V = ti.svd(self.p_F[p])
            #     sig_inv = ti.Matrix.identity(real, self.dim)
            #     for d in ti.static(range(self.dim)):
            #         sig[d, d] = ti.min(ti.max(sig[d, d], 1 - 0.015), 1 + 0.005)  # Plasticity
            #         sig_inv[d, d] = 1 / sig[d ,d]
            #     self.p_Fp[p] = V @ sig_inv @ U.transpose() @ self.p_F[p] @ self.p_Fp[p]
            #     self.p_F[p] = U @ sig @ V.transpose()


            # Loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.grid_v[base + offset] += weight * self.p_mass[p] * self.p_v[p]
                self.grid_m[base + offset] += weight * self.p_mass[p]

    @ti.kernel
    def grid_normalization_and_gravity(self, dt: real):
        self.num_active_grid[None] = 0

        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:  # No need for epsilon here
                self.grid_v[I] = (1 / self.grid_m[I]
                                  ) * self.grid_v[I]  # Momentum to velocity
                self.grid_dv[I] = self.grid_v[I]
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
    def explicit_force(self, dt: real):
        # force is computed more than once in Newton iteration 
        # temporarily set all grid force to zero
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:  # No need for epsilon here
                self.grid_f[I] = ti.Vector.zero(real, self.dim)

        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.p_x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            fx = self.p_x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            dw = [fx-1.5, -2*(fx-1), fx-0.5]

            # # Hardening coefficient: snow gets harder when compressed
            # # h = ti.exp(7 * (1.0 - self.p_Jp[p]))
            # h = ti.exp(7 * (1.0 - self.p_Fp[p].determinant()))
            # mu, la = self.mu_0 * h, self.lambda_0 * h

            mu, la = self.mu_0, self.lambda_0
            # U, sig, V = ti.svd(self.p_F[p])
            # J = self.p_F[p].determinant()
            # P = 2 * mu * (self.p_F[p] - U @ V.T()) @ self.p_F[p].T(
            #     ) + ti.Matrix.identity(real, self.dim) * la * J * (J - 1)
            P = elasticity_first_piola_kirchoff_stress(self.p_F[p], la, mu)
            P = P @ self.p_F_backup[p].transpose()

            vol = self.p_vol[p]

            for offset in ti.static(ti.grouped(self.stencil_range())):
                weight = 1.0
                dN = ti.Vector.zero(real, self.dim)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                if ti.static(self.dim == 2):
                    dN[0] = dw[offset[0]][0]*w[offset[1]][1] * self.inv_dx
                    dN[1] = w[offset[0]][0]*dw[offset[1]][1] * self.inv_dx
                else:
                    dN[0] = dw[offset[0]][0]*w[offset[1]][1]*w[offset[2]][2] * self.inv_dx
                    dN[1] = w[offset[0]][0]*dw[offset[1]][1]*w[offset[2]][2] * self.inv_dx
                    dN[2] = w[offset[0]][0]*w[offset[1]][1]*dw[offset[2]][2] * self.inv_dx


                self.grid_f[base + offset] += - vol * P @ dN                      


    @ti.kernel
    def implicit_update(self, dt:real):
        # for i in range(self.num_active_grid[None]):
        #     dv = ti.Vector([self.data_x[i*2], self.data_x[i*2+1]])
        #     gid = self.dof2idx[i]
        #     self.grid_v[gid[0], gid[1]] += dv     

        for i in range(self.num_active_grid[None]):
            dv = ti.Vector([self.dv[i*2], self.dv[i*2+1]])
            gid = self.dof2idx[i]
            self.grid_v[gid[0], gid[1]] += dv    

    @ti.kernel
    def explicit_update(self, dt:real):
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

    @ti.kernel
    def BuildMatrix(self, dt:real):
        # Build Matrix: Inertial term: dim*N in total
        for i in ti.ndrange(self.num_active_grid[None]):
            for j in range(25):
                self.entryCol[i*25+j] = -1
                self.entryVal[i*25+j] = ti.Matrix.zero(real, self.dim, self.dim)
            gid = self.dof2idx[i]
            m = self.grid_m[gid[0], gid[1]]
            self.entryCol[i*25+12] = i
            self.entryVal[i*25+12] = ti.Matrix.identity(real, self.dim) * m


        # Build Matrix: Loop over all particles
        for I in ti.grouped(self.pid):
            p = self.pid[I]

            vol = self.p_vol[p]
            F = self.p_F[p]
            mu, la = self.mu_0, self.lambda_0
            dPdF = elasticity_first_piola_kirchoff_stress_derivative(F, la, mu)

            
            base = ti.floor(self.p_x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            fx = self.p_x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            dw = [fx-1.5, -2*(fx-1), fx-0.5]

            cnt = 0
            cached_w = []
            cached_node = []
            cached_idx = []
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dN = ti.Vector.zero(real, self.dim)
                dN[0] = dw[offset[0]][0]*w[offset[1]][1] * self.inv_dx
                dN[1] = w[offset[0]][0]*dw[offset[1]][1] * self.inv_dx
                # cached_w.append(F.transpose() @ dN)
                cached_w.append(self.p_F_backup[p].transpose() @ dN)
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


                    dFdX = ti.Matrix.zero(real, self.dim, self.dim)
                    for q in ti.static(range(self.dim)):
                        for v in ti.static(range(self.dim)):
                            dFdX[0,0] += dPdF[self.dim*v+0,self.dim*q+0]*wi[v]*wj[q]
                            dFdX[0,1] += dPdF[self.dim*v+0,self.dim*q+1]*wi[v]*wj[q]
                            dFdX[1,0] += dPdF[self.dim*v+1,self.dim*q+0]*wi[v]*wj[q]
                            dFdX[1,1] += dPdF[self.dim*v+1,self.dim*q+1]*wi[v]*wj[q]
                    dFdX = dFdX * vol * dt * dt


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

        
        # Uncomment this part if no CG projection
        ndof = self.num_active_grid[None]
        for i in range(ndof):
            if self.boundary[i] == 1:
                srt = i*25
                end = srt + 25
                for k in range(srt, end):
                    j = self.entryCol[k]
                    if i == j:
                        self.entryVal[k] = ti.Matrix.identity(real, self.dim)
                    elif not j == -1:
                        self.entryCol[k] = -1
                        self.entryVal[k] = ti.Matrix.zero(real, self.dim, self.dim)  
                        srt2 = j*25
                        end2 = srt2+25
                        for k2 in range(srt2, end2):
                            i2 = self.entryCol[k2]
                            if i2 == i:
                                self.entryCol[k2] = -1
                                self.entryVal[k2] = ti.Matrix.zero(real, self.dim, self.dim)  

    @ti.kernel
    def ComputeResidual(self, dt:real):
        ndof = self.num_active_grid[None]
        d = self.dim
        g = self.gravity[None]
        for i in ti.ndrange(ndof):
            gid = self.dof2idx[i]
            m = self.grid_m[gid]
            f = self.grid_f[gid]

            self.rhs[i*d+0] = self.dv[i*d+0] * m - dt * f[0] - dt * m * g[0]
            self.rhs[i*d+1] = self.dv[i*d+1] * m - dt * f[1] - dt * m * g[1]


        # Uncomment this part if no CG projection
        ndof = self.num_active_grid[None]
        for i in range(ndof):
            if self.boundary[i] == 1:
                for d in ti.static(range(self.dim)):
                    self.rhs[i * self.dim + d] = 0

    @ti.kernel
    def TotalEnergy(self, dt: real) -> real:
        '''
            Compute total energy
        '''
        ee = 0.0
        for p in range(self.n_particles[None]):
        # for I in ti.grouped(self.pid):
        #     p = self.pid[I]
            F = self.p_F[p]
            la, mu = self.lambda_0, self.mu_0
            U, sig, V = ti.svd(F)
            psi = elasticity_energy(sig, la, mu)
            ee += self.p_vol[p] * psi

        ke = 0.0
        for i in range(self.num_active_grid[None]):
            dv = ti.Vector([self.DV[2*i], self.DV[2*i+1]])
            gid = self.dof2idx[i]
            ke += dv.dot(dv) * self.grid_m[gid]

        ge = 0.0
        for i in range(self.num_active_grid[None]):
            dv = ti.Vector([self.DV[2*i], self.DV[2*i+1]])
            gid = self.dof2idx[i]
            ge -= dt * dv.dot(self.gravity[None]) * self.grid_m[gid]

        return ee + ke / 2 + ge

    @ti.kernel
    def checkNewton(self):
        s = 0.0
        for i in range(self.num_active_grid[None] * self.dim):
            s += self.data_x[i] * self.rhs[i]
        print("aaaa", s)
        assert s < 0

    def checkHessian(self):
        print("abcdef")

        ndof = self.num_active_grid[None]
        d = self.dim

        nentry = ndof * d * 25 * 4
        A = scipy.sparse.csr_matrix((self.data_val.to_numpy()[0:nentry], (self.data_row.to_numpy()[0:nentry], self.data_col.to_numpy()[0:nentry])), shape=(ndof*d,ndof*d))

        A[0,0] = self.matrix[0,0]
        # for i in range(10):
        #     print(A[i,i],self.matrix[i,i])
        # print(scipy.sparse.linalg.eigs(A, k=ndof*d-2)[0].real)
        scipy.sparse.save_npz('A.npz', A)


    def SolveLinearSystem(self, dt):
        ndof = self.num_active_grid[None]
        d = self.dim

        self.matrix.prepareColandVal(ndof)
        self.matrix.setFromColandVal(self.entryCol, self.entryVal, ndof)
        # self.matrix.setIdentity(ndof*d)

        self.cgsolver.compute(self.matrix)
        self.cgsolver.setBoundary(self.boundary)
        self.cgsolver.solve(self.rhs)

        for i in range(ndof):
            self.data_x[i*d], self.data_x[i*d+1] = -self.cgsolver.x[i*d], -self.cgsolver.x[i*d+1] 

        # for i in range(ndof):
        #     self.data_x[i*d], self.data_x[i*d+1] = -self.rhs[i*d], -self.rhs[i*d+1]

        # # cg accuracy
        # self.cgsolver.computAp(self.cgsolver.x)
        # for i in range(10):
        #     print(self.rhs[i], self.cgsolver.Ap[i])

        # rhs = np.zeros(ndof*d)
        # rhs = self.rhs.to_numpy()[0:ndof*d]

        # nentry = ndof * d * 25 * 4
        # A = scipy.sparse.csr_matrix((self.data_val.to_numpy()[0:nentry], (self.data_row.to_numpy()[0:nentry], self.data_col.to_numpy()[0:nentry])), shape=(ndof*d,ndof*d))
        # A[0,0] = self.matrix[0,0]
        # # x = scipy.sparse.linalg.spsolve(A, -rhs)
        # x, flag = scipy.sparse.linalg.cg(A, -rhs)
        # assert flag == 0

        # for i in range(ndof):
        #     self.data_x[i*d], self.data_x[i*d+1] = x[i*d], x[i*d+1]


        # self.checkNewton()
        # self.checkHessian() 

    @ti.kernel
    def UpdateState(self, dt:real, alpha:real):
        # Update deformation gradient F = (I + deltat * d(v'))F
        # where v' = v + dv + alpha * ddv
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.p_x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)
            fx = self.p_x[p] * self.inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2
            ]
            dw = [fx-1.5, -2*(fx-1), fx-0.5]
            new_F = ti.Matrix.zero(real, self.dim, self.dim)
            # loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                g_v = self.grid_v[base + offset]
                g_dof = self.grid_idx[base + offset]
                DV = ti.Vector([self.DV[2*g_dof], self.DV[2*g_dof+1]])
                g_v += DV
                weight = 1.0
                dN = ti.Vector.zero(real, self.dim)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                if ti.static(self.dim == 2):
                    dN[0] = dw[offset[0]][0]*w[offset[1]][1] * self.inv_dx
                    dN[1] = w[offset[0]][0]*dw[offset[1]][1] * self.inv_dx
                else:
                    dN[0] = dw[offset[0]][0]*w[offset[1]][1]*w[offset[2]][2] * self.inv_dx
                    dN[1] = w[offset[0]][0]*dw[offset[1]][1]*w[offset[2]][2] * self.inv_dx
                    dN[2] = w[offset[0]][0]*w[offset[1]][1]*dw[offset[2]][2] * self.inv_dx
                new_F += g_v.outer_product(dN)
            self.p_F[p] = (ti.Matrix.identity(real, self.dim) + dt * new_F) @ self.p_F[p]

    @ti.kernel
    def BackupStrain(self):
        for i in range(self.n_particles[None]):
            self.p_F_backup[i] = self.p_F[i]

    @ti.kernel
    def RestoreStrain(self):
        for i in range(self.n_particles[None]):
            self.p_F[i] = self.p_F_backup[i]

    @ti.kernel
    def UpdateDV(self, alpha:real):
        for i in range(self.num_active_grid[None] * self.dim):
            self.DV[i] = self.dv[i] + self.data_x[i] * alpha

    @ti.kernel
    def LineSearch(self, dt:real, alpha:real):
        # self.dv += self.data_x
        for i in range(self.num_active_grid[None] * self.dim):
            self.dv[i] += self.data_x[i] * alpha

    @ti.kernel
    def BuildInitialBoundary(self, dt:real):
        '''
            Set initial guess for newton iteration
        '''
        ndof = self.num_active_grid[None]
        for i in range(ndof):
            if self.boundary[i] == 1:
                self.dv[i*2] = 0
                self.dv[i*2+1] = 0
            else:
                g = self.gravity[None]
                # g = ti.Vector.zero(real, self.dim)
                self.dv[i*2] = g[0]*dt
                self.dv[i*2+1] = g[1]*dt

    @ti.kernel
    def BuildInitial(self, dt:real):
        '''
            Set initial guess for newton iteration
        '''
        # ndof = self.num_active_grid[None]
        # padding = 0.1 / self.dx 
        # for i in range(ndof):
        #     gid = self.dof2idx[i]
        #     old_v = self.grid_v[gid]
        #     vi = ti.Vector([old_v[0], old_v[1]])
        #     if gid[0] < padding or gid[1] < padding:
        #         self.boundary[i] = 1
        #         vi[1] = 0.0
        #         vi = vi - old_v
        #         self.dv[i*2] = vi[0]
        #         self.dv[i*2+1] = vi[1] 
        #     else:
        #         self.dv[i*2]=0
        #         self.dv[i*2+1]=self.gravity[None][1]*self.dt

        ndof = self.num_active_grid[None]
        for i in range(ndof):
            gid = self.dof2idx[i]
            x = gid * self.dx
            old_v = self.grid_v[gid]
            # flag, vi = self.analytic_collision[0](x, old_v)
            # flag = 0
            vi = ti.Vector([old_v[0], old_v[1]])
            for k in ti.static(range(len(self.analytic_collision))):
                f, vn = self.analytic_collision[k](x, vi)
                if f == 1:
                    # flag = 1
                    self.boundary[i] = 1
                    vi = vn
            if self.boundary[i] == 1:
                vi = vi - old_v
                self.dv[i*2] = vi[0]
                self.dv[i*2+1] = vi[1]
            else:
                g = self.gravity[None]
                # g = ti.Vector.zero(real, self.dim)
                self.dv[i*2] = g[0]*dt
                self.dv[i*2+1] = g[1]*dt

    @ti.kernel
    def computeScaledNorm(self)->real:
        result = 0.0
        for i in range(self.num_active_grid[None]):
            result += (self.rhs[2*i] ** 2 + self.rhs[2*i+1] ** 2)/(self.nodeCNTol[i] ** 2)
        return result


    def implicit_newton(self, dt):
        # perform one full newton
        self.evaluatePerNodeCNTolerance(1e-7, dt)

        self.BackupStrain()

        self.data_x.fill(0)
        self.rhs.fill(0)
        self.dv.fill(0)
        self.DV.fill(0)
        self.BuildInitialBoundary(dt)

        # Newton iteration

        max_iter_newton = 150
        for iter in range(max_iter_newton):
            print('Newton iter = ', iter)
            self.RestoreStrain()
            self.rhs.fill(0)
            self.UpdateDV(0.0)
            self.UpdateState(dt, 0)
            E0 = self.TotalEnergy(dt)
            print("E0 = ", E0)
            self.explicit_force(dt) 
            self.ComputeResidual(dt) # Compute g

            ndof = self.num_active_grid[None]
            rnorm = np.linalg.norm(self.rhs.to_numpy()[0:ndof*self.dim])
            print("norm", rnorm)
            scaledNorm = self.computeScaledNorm()
            print("snorm", scaledNorm, self.num_active_grid[None])
            # if rnorm < 1e-8:
            if scaledNorm < self.num_active_grid[None]:
                print("Newton finish in", iter, ", residual =", rnorm)
                break

            self.BuildMatrix(dt) # Compute H
            self.data_col.fill(0)
            self.data_row.fill(0)
            self.build_T()

            self.SolveLinearSystem(dt)  # solve dx = H^(-1)g

            alpha = 1.0
            for _ in range(15):
                self.RestoreStrain()
                self.UpdateDV(alpha)
                self.UpdateState(dt, alpha)
                E = self.TotalEnergy(dt)
                # print("alpha=",alpha,"E=",E)
                if E <= E0:
                    break
                alpha /= 2
            if alpha == 1/2**15:
                print("Line Search ERROR!!! Check the direction!!")
                alpha = 0.0

                # self.checkHessian()
                break
            print(alpha, E)
            self.LineSearch(dt, alpha)

            self.RestoreStrain() 
      
        print("aaaa", iter)
        if iter == max_iter_newton - 1:
            print("Newton Max iteration reached!")
        
        self.implicit_update(dt)
        # self.TotalEnergy()
        self.RestoreStrain() 

    def implicit_newton2(self, dt):
        # perform one full newton
        self.ddv.fill(0)
        self.rhs.fill(0)
        self.dv.fill(0)
        # self.boundary.fill(0)
        # self.BuildInitial(dt)
        self.BuildInitialBoundary(dt)
        # Newton iteration
        self.BackupStrain()
        for iter in range(1):
            with Timer("Build System"):
                self.explicit_force(dt) 
                self.ComputeResidual(dt) # Compute g
                self.BuildMatrix(dt) # Compute H
                self.data_col.fill(0)
                self.data_row.fill(0)
                self.build_T()
            with Timer("Solve System"):
                self.SolveLinearSystem(dt)  # solve dx = H^(-1)g

            self.LineSearch(dt, 1.0)

            Timer_Print()     
        self.implicit_update(dt)

    # @ti.kernel
    # def grid_collision(self, dt: real):
    #     padding = 3
    #     for I in ti.grouped(self.grid_m):
    #         for d in ti.static(range(self.dim)):
    #             if False:
    #                 if I[d] < -self.grid_size // 2 + padding and self.grid_v[
    #                         I][d] < 0:
    #                     self.grid_v[I][d] = 0  # Boundary conditions
    #                 if I[d] >= self.grid_size // 2 - padding and self.grid_v[
    #                         I][d] > 0:
    #                     self.grid_v[I][d] = 0
    #             else:
    #                 if I[d] < padding and self.grid_v[I][d] < 0:
    #                     self.grid_v[I][d] = 0  # Boundary conditions
    #                 if I[d] >= self.res[d] - padding and self.grid_v[I][
    #                         d] > 0:
    #                     self.grid_v[I][d] = 0


    @ti.kernel
    def g2p(self, dt: real):
        ti.block_dim(256)
        ti.cache_shared(*self.grid_v.entries)
        ti.cache_shared(*self.grid_dv.entries)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.p_x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)
            fx = self.p_x[p] * self.inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2
            ]
            dw = [fx-1.5, -2*(fx-1), fx-0.5]
            new_v = ti.Vector.zero(real, self.dim)
            v_pic = ti.Vector.zero(real, self.dim)
            v_flip = self.p_v[p]
            new_C = ti.Matrix.zero(real, self.dim, self.dim)
            new_F = ti.Matrix.zero(real, self.dim, self.dim)
            # loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(float) - fx
                g_v = self.grid_v[base + offset]
                weight = 1.0
                dN = ti.Vector.zero(real, self.dim)
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
                v_pic += weight * g_v
                v_flip += weight * (g_v-self.grid_dv[base + offset])
                # v_flip -= weight * (self.grid_dv[base + offset])
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
                new_F += g_v.outer_product(dN)
            self.p_v[p], self.p_C[p] = new_v, new_C
            self.p_v[p] = v_flip * 0.95 + v_pic * 0.05
            # if v_flip.norm() > 1e-6:
            #     print(v_flip.norm())
            self.p_x[p] += dt * v_pic  # advection
            self.p_F[p] = (ti.Matrix.identity(real, self.dim) + dt * new_F)@ self.p_F[p]


    @ti.kernel
    def update_volume(self):
        # ti.block_dim(256)
        # ti.cache_shared(*self.grid_v.entries)
        # ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.p_x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)
            fx = self.p_x[p] * self.inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2
            ]

            new_density = 0.0
            # loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_density += weight * self.grid_m[base + offset] / self.dx ** self.dim
            self.p_rho[p] = new_density
            self.p_vol[p] = self.p_mass[p] / self.p_rho[p]

    @ti.kernel
    def evaluatePerNodeCNTolerance(self, eps:real, dt:real):
        for i in range(self.num_active_grid[None]):
            self.nodeCNTol[i] = 0.0

        for I in ti.grouped(self.pid):
            p = self.pid[I]

            F = self.p_F[p]
            mu, la = self.mu_0, self.lambda_0
            dPdF = elasticity_first_piola_kirchoff_stress_derivative(F, la, mu)


            base = ti.floor(self.p_x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            fx = self.p_x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

            # Loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                gid = self.grid_idx[base + offset]    
                self.nodeCNTol[gid] += weight * self.p_mass[p] * dPdF.norm()
        
        for i in range(self.num_active_grid[None]):
            self.nodeCNTol[i] *= eps * 8 * self.dx * dt / self.grid_m[self.dof2idx[i]]
            # self.nodeCNTol[i] *= eps * 24 * self.dx * self.dx * dt / self.grid_m[self.dof2idx[i]]


    def advanceOneStepExplicit(self, dt):
        self.grid.deactivate_all()
        self.build_pid()
        self.p2g(dt)
        self.grid_normalization_and_gravity(dt)

        # if self.total_step[None] == 0:
        #     print("Correct")
        #     self.update_volume()
        self.BackupStrain() # TODO: 
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
        # self.explicit_force(dt)
        
        self.boundary.fill(0)
        for p in self.grid_collidable_objects:
            p(dt)
        
        self.implicit_newton(dt)

        self.g2p(dt)

    def step(self, frame_dt, print_stat=False):
        frame_done = False
        frame_time = 0.0
        # self.simulation_info()
        while not frame_done:
            dt = self.dt

            max_v=self.getMaxVelocity()
            print("max velocity:  ", max_v)
            # cfl = 0.1
            # max_dt = 0.005
            # min_dt = 1e-6
            cfl = 0.6
            max_dt = 0.04
            min_dt = 1e-6
            dt=ti.max(min_dt,ti.min(max_dt,cfl*self.dx/ti.max(max_v,1e-2)))


            if frame_dt - frame_time < dt * 1.001:
                frame_done = True
                dt = frame_dt - frame_time
            else:
                if frame_dt - frame_time < 2 * dt:
                    dt = (frame_dt - frame_time) / 2.0


            # self.advanceOneStepExplicit(dt)
            self.advanceOneStepNewton(dt)

            
            frame_time += dt
            self.total_step[None] += 1
            print(int((frame_time / frame_dt)*1000)/10, "%         dt = ", dt)
            



    ########### MPM functions ###########
    @ti.kernel
    def getMaxVelocity(self)-> ti.f32:
        max_v = 0.0
        for i in range(self.n_particles[None]):
            r = self.p_v[i].norm_sqr()
            ti.atomic_max(max_v, r)
        return ti.sqrt(max_v)


    ########### MPM Set parameters ###########
    def setLameParameter(self, E, nu):
        self.mu_0, self.lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

    def setGravity(self, g):
        assert isinstance(g, (tuple, list))
        assert len(g) == self.dim
        self.gravity[None] = g


    ########### Sample MPM Particle Cloud ###########

    @ti.func
    def seed_particle(self, i, pos, vel):
        self.p_x[i] = pos
        self.p_v[i] = vel
        self.p_F[i] = ti.Matrix.identity(real, self.dim)

        self.p_Jp[i] = 1
        self.p_vol[i] = self.dx**self.dim
        self.p_rho[i] = 1000
        self.p_mass[i] = self.p_vol[i] * self.p_rho[i]
        # self.p_vol = 
        # self.p_rho = 
        # self.p_mass = 

    def add_cube(self, min_corner, max_corner, num_particles = 20000):
        a = ti.Vector(min_corner)
        b = ti.Vector(max_corner)
        b = b - a

        self.sample_cube(a, b, num_particles)

    @ti.kernel
    def sample_cube(self, lower_corner: ti.template(), cube_size: ti.template(), new_particles: ti.i32):
        # new_particles = 20000
        area = 1.0
        for k in ti.static(range(self.dim)):
            area *= cube_size[k]
        area_per_particle = area / new_particles
        for i in range(self.n_particles[None], self.n_particles[None] + new_particles):
            x = ti.Vector.zero(real, self.dim)
            for k in ti.static(range(self.dim)):
                x[k] = lower_corner[k] + ti.random()*cube_size[k]
            # self.seed_particle(i, x, ti.Vector.zero(real, self.dim))
            self.p_x[i] = x
            self.p_v[i] = ti.Vector.zero(real, self.dim)
            self.p_F[i] = ti.Matrix.identity(real, self.dim)
            self.p_Fp[i] = ti.Matrix.identity(real, self.dim)
            self.p_Jp[i] = 1.0
            self.p_vol[i] = area_per_particle
            self.p_rho[i] = 2
            self.p_mass[i] = self.p_vol[i] * self.p_rho[i]
        self.n_particles[None] = self.n_particles[None] + new_particles


    @ti.kernel
    def copy_dynamic_nd(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in self.p_x:
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]

    def particle_info(self):
        np_x = np.ndarray((self.n_particles[None], self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_x, self.p_x)
        np_v = np.ndarray((self.n_particles[None], self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_v, self.p_v)

        return {
            'position': np_x,
            'velocity': np_v,
        }

    def simulation_info(self):
        i = 0
        print(self.p_mass[i], self.p_vol[i], self.p_x[i][0], self.p_x[i][1], \
            self.p_v[i][0], self.p_v[i][1], self.p_F[i][0,0], self.p_F[i][0,1], \
                self.p_F[i][1,0], self.p_F[i][1,1])

    ########### Analytic Collision Objects ###########
  
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
                v = ti.Vector.zero(real, self.dim)
            return flag, v
            # normal_component = n.dot(v)
            # v = v - n * normal_component
            # return v
            # if x < padding

        self.analytic_collision.append(get_velocity)

        @ti.kernel
        def collide(dt: real):
            for I in ti.grouped(self.grid_m):
                offset = I * self.dx - ti.Vector(point)
                n = ti.Vector(normal)
                if offset.dot(n) < 0:
                    # self.grid_dv[I] = ti.Vector.zero(real, self.dim)
                    self.grid_v[I] = ti.Vector.zero(real, self.dim)

        self.grid_collidable_objects.append(collide) 


    @ti.kernel
    def test(self):
        pass

    def add_analytic_box(self, min_corner, max_corner, rotation = 0.0):

        min_corner = ti.Vector(min_corner)
        max_corner = ti.Vector(max_corner)

        b = (min_corner + max_corner) / 2
        half_edge = (max_corner - min_corner) / 2

        theta = rotation
        R = ti.Matrix([[ti.cos(theta), -ti.sin(theta)],[ti.sin(theta), ti.cos(theta)]])



        @ti.func
        def signedDistance(x):
            xx = R.transpose() @ (x - b)
            d = ti.Vector.zero(real, self.dim)
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
                v = ti.Vector.zero(real, self.dim)
            return flag, v


        @ti.kernel
        def gridCollision(dt: real):
            for I in ti.grouped(self.grid_m):
                x = I * self.dx
                if signedDistance(x) < 0:
                    self.grid_v[I] = ti.Vector.zero(real, self.dim)
                    gid = self.grid_idx[I]
                    self.boundary[gid] = 1
            
        self.analytic_collision.append(particleCollision)
        self.grid_collidable_objects.append(gridCollision)


    def load_state(self):
        filename = 'test1.txt'
        with open(filename,"r") as f:
            for i in range(20000):
                line = f.readline().split()
                data = [float(x) for x in line]
                # print(data)
                self.p_mass[i] = data[0]
                self.p_vol[i] = data[1]
                self.p_x[i] = ti.Vector([data[2], data[3]])
                self.p_v[i] = ti.Vector([data[4], data[5]])
                self.p_F[i] = ti.Matrix([[data[6], data[7]],[data[8], data[9]]])