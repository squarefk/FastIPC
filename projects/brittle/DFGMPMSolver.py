import taichi as ti
import numpy as np
import math
from common.utils.timer import *

@ti.data_oriented
class DFGMPMSolver:
    #define structs here
    # Surface boundary conditions

    # Stick to the boundary
    surfaceSticky = 0
    # Slippy boundary
    surfaceSlip = 1
    # Slippy and free to separate
    surfaceSeparate = 2

    surfaces = {
        'STICKY': surfaceSticky,
        'SLIP': surfaceSlip,
        'SEPARATE': surfaceSeparate
    }

    #define constructor here
    def __init__(self, endFrame, fps, dt, dx, E, nu, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThresholds, useFrictionalContact, frictionCoefficient = 0.0, verbose = False, useAPIC = False, flipPicRatio = 0.0, useRankineDamage = False, cf = 2000.0, sigmaCrit = 1.0, Gf = 1.0):
        
        #Simulation Parameters
        self.endFrame = endFrame
        self.fps = fps
        self.frameDt = 1.0 / fps
        self.dt = dt
        self.numSubsteps = int(self.frameDt // dt)
        self.dim = len(vertices[0])
        self.cfl = cfl
        self.ppc = ppc
        self.outputPath = outputPath
        self.outputPath2 = outputPath2
        self.vertices = vertices
        self.particleCounts = np.array(particleCounts)
        self.initialVelocity = np.array(initialVelocity)
        self.pMasses = np.array(particleMasses)
        self.pVolumes = np.array(particleVolumes)
        self.numObjects = len(particleCounts) #how many objects are we modeling
        self.numParticles = len(vertices)
        self.surfaceThresholds = np.array(surfaceThresholds)
        self.dx = dx
        self.invDx = 1.0 / dx
        self.nGrid = ti.ceil(self.invDx)
        self.E = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))
        self.la = E * nu / ((1+nu) * (1 - 2 * nu))
        self.gravMag = gravity
        self.useFrictionalContact = useFrictionalContact
        self.verbose = verbose
        self.useAPIC = useAPIC
        self.flipPicRatio = flipPicRatio #default to 0 which means full PIC
        if flipPicRatio < 0.0 or flipPicRatio > 1.0:
            raise ValueError('flipPicRatio must be between 0 and 1')
        self.gridPostProcess = [] #hold function callbacks for post processing velocity

        #Damage Parameters
        self.useRankineDamage = useRankineDamage
        self.sigmaCrit = sigmaCrit
        self.l0 = 0.5 * dx #as usual, close to the sqrt(2) * dx that they use
        self.Gf = Gf
        HsBar = (sigmaCrit * sigmaCrit) / (2 * E * Gf)
        self.Hs = (HsBar * self.l0) / (1 - (HsBar * self.l0))
        self.cf = cf
        self.timeToFail = self.cf / self.l0

        #Neighbor Search Variables - NOTE: if these values are too low, we get data races!!! Try to keep these as high as possible (RIP to ur RAM)
        self.maxNeighbors = 1024
        self.maxPPC = 1024

        #DFG Parameters
        self.rp = (3*(dx**2))**0.5 if self.dim == 3 else (2*(dx**2))**0.5 #set rp based on dx (this changes if dx != dy)
        self.maxParticlesInfluencingGridNode = self.ppc * self.maxPPC #2d = 4*ppc, 3d = 8*ppc
        self.dMin = 0.25
        self.fricCoeff = frictionCoefficient
        self.epsilon_m = 0.0001
        #self.surfaceThreshold = surfaceThreshold 
        self.st = ti.field(dtype=float, shape=self.numParticles) #now we can have different thresholds for different objects and particle distributions!
        
        #Explicit MPM Fields
        self.x = ti.Vector.field(2, dtype=float, shape=self.numParticles) # position
        self.v = ti.Vector.field(2, dtype=float, shape=self.numParticles) # velocity
        self.C = ti.Matrix.field(2, 2, dtype=float, shape=self.numParticles) # affine velocity field
        self.F = ti.Matrix.field(2, 2, dtype=float, shape=self.numParticles) # deformation gradient
        self.material = ti.field(dtype=int, shape=self.numParticles) # material id
        self.mp = ti.field(dtype=float, shape=self.numParticles) # particle masses
        self.Vp = ti.field(dtype=float, shape=self.numParticles) # particle volumes
        self.Jp = ti.field(dtype=float, shape=self.numParticles) # plastic deformation
        self.grid_v = ti.Vector.field(2, dtype=float, shape=(self.nGrid, self.nGrid, 2)) # grid node momentum/velocity, store two vectors at each grid node (for each field)
        self.grid_vn = ti.Vector.field(2, dtype=float, shape=(self.nGrid, self.nGrid, 2)) # use this to store grid v_i^n so we can use this for FLIP
        self.grid_m = ti.Vector.field(2, dtype=float, shape=(self.nGrid, self.nGrid)) # grid node mass is nGrid x nGrid x 2, each grid node has a mass for each field
        self.gravity = ti.Vector.field(2, dtype=float, shape=())

        #DFG Fields
        self.Dp = ti.field(dtype=float, shape=self.numParticles) #particle damage
        self.sp = ti.field(dtype=int, shape=self.numParticles) #particle surface boolean int -- 1 = surface particle, 0 = interior particle
        self.particleDG = ti.Vector.field(2, dtype=float, shape=self.numParticles) #keep track of particle damage gradients
        self.gridDG = ti.Vector.field(2, dtype=float, shape=(self.nGrid, self.nGrid)) #grid node damage gradients
        self.maxHelperCount = ti.field(int)
        self.maxHelper = ti.field(int)
        ti.root.dense(ti.ij, (self.nGrid,self.nGrid)).place(self.maxHelperCount) #maxHelperCount is nGrid x nGrid and keeps track of how many candidates we have for the new nodeDG maximum
        ti.root.dense(ti.ij, (self.nGrid,self.nGrid)).dense(ti.k, self.maxParticlesInfluencingGridNode).place(self.maxHelper) #maxHelper is nGrid x nGrid x maxParticlesInfluencingGridNode
        self.gridSeparability = ti.Vector.field(4, dtype=float, shape=(self.nGrid, self.nGrid)) # grid separability is nGrid x nGrid x 4, each grid node has a seperability condition for each field and we need to add up the numerator and denominator
        self.gridMaxDamage = ti.Vector.field(2, dtype=float, shape=(self.nGrid, self.nGrid)) # grid max damage is nGrid x nGrid x 2, each grid node has a max damage from each field
        self.separable = ti.field(dtype=int, shape=(self.nGrid,self.nGrid)) # whether grid node is separable or not
        self.grid_n = ti.Vector.field(2, dtype=float, shape=(self.nGrid, self.nGrid, 2)) # grid node normals for two field nodes, store two vectors at each grid node (one for each field)
        self.grid_f = ti.Vector.field(2, dtype=float, shape=(self.nGrid, self.nGrid, 2)) # grid node forces for two field nodes, store two vectors at each grid node (one for each field)

        #Active Fields
        self.activeFields = ti.field(int)
        self.activeFieldsCount = ti.field(int)
        ti.root.dense(ti.ij, (self.nGrid,self.nGrid)).dense(ti.k, 2).dense(ti.l, self.maxParticlesInfluencingGridNode).place(self.activeFields) #activeFields is nGrid x nGrid x 2 x numParticlesMappingToGridNode
        ti.root.dense(ti.ij, (self.nGrid,self.nGrid)).dense(ti.k, 2).place(self.activeFieldsCount) #activeFieldsCount is nGrid x nGrid x 2 to hold counters for the active field lists
        self.particleAF = ti.Vector.field(9, dtype=int, shape=self.numParticles) #store which activefield each particle belongs to for the 9 grid nodes it maps to

        #Neighbor Search Fields
        self.gridNumParticles = ti.field(int)      #track number of particles in each cell using cell index
        self.backGrid = ti.field(int)              #background grid to map grid cells to a list of particles they contain
        self.particleNumNeighbors = ti.field(int)  #track how many neighbors each particle has
        self.particleNeighbors = ti.field(int)     #map a particle to its list of neighbors
        #Shape the neighbor fields
        self.gridShape = ti.root.dense(ti.ij, (self.nGrid,self.nGrid))
        self.gridShape.place(self.gridNumParticles) #gridNumParticles is nGrid x nGrid
        self.gridShape.dense(ti.k, self.maxPPC).place(self.backGrid) #backGrid is nGrid x nGrid x maxPPC
        #NOTE: backgrid uses nGrid x nGrid even though dx != rp, but rp is ALWAYs larger than dx so it's okay!
        self.particleShape = ti.root.dense(ti.i, self.numParticles)
        self.particleShape.place(self.particleNumNeighbors) #particleNumNeighbors is nParticles x 1
        self.particleShape.dense(ti.j, self.maxNeighbors).place(self.particleNeighbors) #particleNeighbors is nParticles x maxNeighbors

    ##########

    #Constitutive Model
    @ti.func 
    def kirchoff_FCR(self, F, R, J, mu, la):
        #compute Kirchoff stress using FCR elasticity
        return 2 * mu * (F - R) @ F.transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1) #compute kirchoff stress for FCR model (remember tau = P F^T)

    @ti.func
    def kirchoff_NeoHookean(self, F, J, mu, la):
        #compute Kirchoff stress using compressive NeoHookean elasticity (Eq 22. in Homel2016 but Kirchoff stress)
        return J * ((((la * (ti.log(J) / J)) - (mu / J)) * ti.Matrix.identity(float, 2)) + ((mu / J) * F @ F.transpose()))

    ##########

    #Neighbor Search Routines
    @ti.func
    def backGridIdx(self, x):
        #compute int vector of backGrid indeces (recall the grid here )
        return int(x/self.rp)

    @ti.func
    def isInGrid(self, x):
        return 0 <= x[0] and x[0] < self.nGrid and 0 <= x[1] and x[1] < self.nGrid

    ##########

    #DFG Function Evaluations
    @ti.func
    def computeOmega(self, rBar): 
        #compute kernel function
        return 1 - (3 * (rBar**2)) + (2 * (rBar**3)) if (rBar >= 0 and rBar <= 1) else 0

    @ti.func
    def computeOmegaPrime(self, rBar): 
        #compute kernel function derivative
        return 6*(rBar**2 - rBar) if (rBar >= 0 and rBar <= 1) else 0

    @ti.func
    def computeRBar(self, x, xp): 
        #compute normalized distance
        return (x-xp).norm() / self.rp

    @ti.func
    def computeRBarGrad(self, x, xp): 
        #compute gradient of normalized distance
        return (x - xp) * (1 / (self.rp * (x-xp).norm()))

    ##########

    #Simulation Routines
    @ti.kernel
    def reinitializeStructures(self):
        #re-initialize grid quantities
        for i, j in self.grid_m:
            self.grid_v[i, j, 0] = [0, 0] #field 1 vel
            self.grid_v[i, j, 1] = [0, 0] #field 2 vel
            self.grid_vn[i, j, 0] = [0, 0] #field 1 vel v_i^n
            self.grid_vn[i, j, 1] = [0, 0] #field 2 vel v_i^n
            self.grid_n[i, j, 0] = [0, 0] #field 1 normal
            self.grid_n[i, j, 1] = [0, 0] #field 2 normal
            self.grid_f[i, j, 0] = [0, 0] #f1 nodal force
            self.grid_f[i, j, 1] = [0, 0] #f2 nodal force
            self.grid_m[i, j] = [0, 0] #stacked to contain mass for each field
            self.gridSeparability[i, j] = [0, 0, 0, 0] #stackt fields, and we use the space to add up the numerator and denom for each field
            self.gridMaxDamage[i, j] = [0, 0] #stackt fields
            self.gridDG[i, j] = [0, 0] #reset grid node damage gradients
            self.separable[i,j] = -1 #-1 for only one field, 0 for not separable, and 1 for separable
        
        #Clear neighbor look up structures as well as maxHelperCount and activeFieldsCount
        for I in ti.grouped(self.gridNumParticles):
            self.gridNumParticles[I] = 0
        for I in ti.grouped(self.particleNeighbors):
            self.particleNeighbors[I] = -1
        for I in ti.grouped(self.maxHelperCount):
            self.maxHelperCount[I] = 0
        for I in ti.grouped(self.activeFieldsCount):
            self.activeFieldsCount[I] = 0
        for I in ti.grouped(self.particleAF):
            self.particleAF[I] = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    
    @ti.kernel
    def backGridSort(self):
        #Sort particles into backGrid
        for p in range(self.numParticles):
            cell = self.backGridIdx(self.x[p]) #grab cell idx (vector of ints)
            offs = ti.atomic_add(self.gridNumParticles[cell], 1) #atomically add one to our grid cell's particle count NOTE: returns the OLD value before add
            self.backGrid[cell, offs] = p #place particle idx into the grid cell bucket at the correct place in the cell's neighbor list (using offs)

    @ti.kernel
    def particleNeighborSorting(self):
        #Sort into particle neighbor lists
        #See https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py for reference
        for p_i in range(self.numParticles):
            pos = self.x[p_i]
            cell = self.backGridIdx(pos)
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1,2),(-1,2)))):
                cellToCheck = cell + offs
                if self.isInGrid(cellToCheck):
                    for j in range(self.gridNumParticles[cellToCheck]): #iterate over all particles in this cell
                        p_j = self.backGrid[cellToCheck, j]
                        if nb_i < self.maxNeighbors and p_j != p_i and (pos - self.x[p_j]).norm() < self.rp:
                            self.particleNeighbors[p_i, nb_i] = p_j
                            nb_i += 1
            self.particleNumNeighbors[p_i] = nb_i

    @ti.kernel
    def surfaceDetection(self):
        #Surface Detection (set sp values before DGs!!!)
        #NOTE: Set whether this particle is a surface particle or not by comparing the kernel sum, S(x), against a user threshold
        for p in range(self.numParticles):
            S = 0.0
            pos = self.x[p]
            for i in range(self.particleNumNeighbors[p]):
                xp_i = self.particleNeighbors[p, i] #index of curr neighbor
                xp = self.x[xp_i] #grab neighbor position
                rBar = self.computeRBar(pos, xp)
                S += self.computeOmega(rBar)
            if(S <= self.st[p]):
                self.sp[p] = 1
            elif(self.sp[p] != 1):
                self.sp[p] = 0

    @ti.kernel
    def computeParticleDG(self):
        #Compute DG for all particles and for all grid nodes 
        # NOTE: grid node DG is based on max of mapped particle DGs, in this loop we simply create a list of candidates, then we will take max after
        for p in range(self.numParticles):
            pos = self.x[p]
            DandS = ti.Vector([0.0, 0.0]) #hold D and S in here (no temporary atomic scalar variables in taichi...)
            nablaD = ti.Vector([0.0, 0.0])
            nablaS = ti.Vector([0.0, 0.0])
            for i in range(self.particleNumNeighbors[p]): #iterate over neighbors of particle p

                xp_i = self.particleNeighbors[p, i] #index of curr neighbor
                xp = self.x[xp_i] #grab neighbor position
                rBar = self.computeRBar(pos, xp)
                rBarGrad = self.computeRBarGrad(pos, xp)
                omega = self.computeOmega(rBar)
                omegaPrime = self.computeOmegaPrime(rBar)

                #Add onto the sums for D, S, nablaD, nablaS
                maxD = max(self.Dp[xp_i], self.sp[xp_i])
                deltaDS = ti.Vector([(maxD * omega), omega])
                DandS += deltaDS
                nablaD += (maxD * omegaPrime * rBarGrad)
                nablaS += (omegaPrime * rBarGrad)

            nablaDBar = (nablaD * DandS[1] - DandS[0] * nablaS) / (DandS[1]**2) #final particle DG
            self.particleDG[p] = nablaDBar #store particle DG

            #Now iterate over the grid nodes particle p maps to to set Di!
            base = (pos * self.invDx - 0.5).cast(int)
            for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                gridIdx = base + offset
                currGridDG = self.gridDG[gridIdx] # grab current grid Di

                if(nablaDBar.norm() > currGridDG.norm()): #save this particle's index as a potential candidate for new maximum
                    offs = ti.atomic_add(self.maxHelperCount[gridIdx], 1) #this lets us keep a dynamically sized list by tracking the index
                    self.maxHelper[gridIdx, offs] = p #add this particle index to the list!

    @ti.kernel
    def computeGridDG(self):
        #Now iterate over all active grid nodes and compute the maximum of candidate DGs
        for i, j in self.maxHelperCount:
            currMaxDG = self.gridDG[i,j] # grab current grid Di
            currMaxNorm = currMaxDG.norm()

            for k in range(self.maxHelperCount[i,j]):
                p_i = self.maxHelper[i,j,k] #grab particle index
                candidateDG = self.particleDG[p_i]
                if(candidateDG.norm() > currMaxNorm):
                    currMaxNorm = candidateDG.norm()
                    currMaxDG = candidateDG

            self.gridDG[i,j] = currMaxDG #set to be the max we found

    @ti.kernel
    def massP2G(self):
        # P2G for mass, set active fields, and compute separability conditions
        for p in range(self.numParticles): 
            
            #for particle p, compute base index
            base = (self.x[p] * self.invDx - 0.5).cast(int)
            fx = self.x[p] * self.invDx - base.cast(float)
            
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            #P2G for mass, set active fields, and compute separability conditions
            for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                gridIdx = base + offset
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]

                maxD = max(self.Dp[p], self.sp[p]) #use max of damage and surface particle markers so we detect green case correctly

                #Set Active Fields for each grid node! 
                if self.particleDG[p].dot(self.gridDG[gridIdx]) >= 0:
                    offs = ti.atomic_add(self.activeFieldsCount[gridIdx, 0], 1) #this lets us keep a dynamically sized list by tracking the index
                    self.activeFields[gridIdx, 0, offs] = p #add this particle index to the list!
                    self.grid_m[gridIdx][0] += weight * self.mp[p] #add mass to active field for this particle
                    self.gridSeparability[gridIdx][0] += weight * maxD * self.mp[p] #numerator, field 1
                    self.gridSeparability[gridIdx][2] += weight * self.mp[p] #denom, field 1
                    self.particleAF[p][i*3 + j] = 0 #set this particle's AF to 0 for this grid node
                else:
                    offs = ti.atomic_add(self.activeFieldsCount[gridIdx, 1], 1)
                    self.activeFields[gridIdx, 1, offs] = p
                    self.grid_m[gridIdx][1] += weight * self.mp[p] #add mass to active field for this particle
                    self.gridSeparability[gridIdx][1] += weight * maxD * self.mp[p] #numerator, field 2
                    self.gridSeparability[gridIdx][3] += weight * self.mp[p] #denom, field 2
                    self.particleAF[p][i*3 + j] = 1 #set this particle's AF to 1 for this grid node

    @ti.kernel
    def computeSeparability(self):
        #Iterate grid nodes to compute separability condition and maxDamage (both for each field)
        for i, j in self.gridSeparability:
            gridIdx = ti.Vector([i, j])

            #Compute seperability for field 1 and store as idx 0
            self.gridSeparability[gridIdx][0] /= self.gridSeparability[gridIdx][2] #divide numerator by denominator

            #Compute seperability for field 2 and store as idx 1
            self.gridSeparability[gridIdx][1] /= self.gridSeparability[gridIdx][3] #divide numerator by denominator

            #Compute maximum damage for grid node active fields
            max1 = 0.0
            max2 = 0.0
            for k in range(self.activeFieldsCount[gridIdx, 0]):
                p_i = self.activeFields[gridIdx, 0, k]
                p_d = max(self.Dp[p_i], self.sp[p_i]) #TODO is this right?
                if p_d > max1:
                    max1 = p_d
            for k in range(self.activeFieldsCount[gridIdx, 1]):
                p_i = self.activeFields[gridIdx, 1, k]
                p_d = max(self.Dp[p_i], self.sp[p_i])
                if p_d > max2:
                    max2 = p_d
            self.gridMaxDamage[gridIdx][0] = max1
            self.gridMaxDamage[gridIdx][1] = max2

            # if(i * self.nGrid + j == 1151):
            #     print("node 1151 field one count:", self.activeFieldsCount[gridIdx, 0], "field two count:", self.activeFieldsCount[gridIdx, 1])

            #NOTE: separable[i,j] = -1 for one field, 0 for two non-separable fields, and 1 for two separable fields
            if self.grid_m[i,j][0] > 0 and self.grid_m[i,j][1] > 0:
                minSep = self.gridSeparability[i,j][0] if self.gridSeparability[i,j][0] < self.gridSeparability[i,j][1] else self.gridSeparability[i,j][1]
                maxMax = self.gridMaxDamage[i,j][0] if self.gridMaxDamage[i,j][0] > self.gridMaxDamage[i,j][1] else self.gridMaxDamage[i,j][1]
                if maxMax == 1.0 and minSep > self.dMin:
                    self.separable[i,j] = 1
                else:
                    self.separable[i,j] = 0

    @ti.kernel
    def momentumP2GandForces(self):
        # P2G and Internal Grid Forces
        for p in range(self.numParticles):
            
            #for particle p, compute base index
            base = (self.x[p] * self.invDx - 0.5).cast(int)
            fx = self.x[p] * self.invDx - base.cast(float)
            
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]

            U, sig, V = ti.svd(self.F[p])
            J = 1.0

            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                self.Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            
            #Compute Kirchoff Stress
            kirchoff = self.kirchoff_FCR(self.F[p], U@V.transpose(), J, self.mu, self.la)

            #----DAMAGE ROUTINES BEGIN----
            if(self.useRankineDamage):
                #Get cauchy stress and its eigenvalues and eigenvectors
                kirchoff = self.kirchoff_NeoHookean(self.F[p], J, self.mu, self.la) #compute kirchoff stress using the NH model from homel2016
                U2, sig2, V2 = ti.svd(kirchoff / J) #divide out J to get cauchy stress
                maxEigVal = sig2[0,0]
                for d in ti.static(range(2)):
                    if sig2[d,d] > maxEigVal: maxEigVal = sig2[d,d] #compute max eigenvalue 
                
                #Update Particle Damage
                # if(maxEigVal > self.sigmaCrit):
                #     dNew = (1 + self.Hs) * (1 - (self.sigmaCrit / maxEigVal))
                #     self.Dp[p] = dNew

                #     #Reconstruct tensile scaled Cauchy stress
                #     for d in ti.static(range(2)):
                #         if sig[d,d] > 0: sig[d,d] *= (1 - dNew) #scale tensile eigenvalues using the new damage
                dNew = self.Dp[p] + (self.dt / self.timeToFail)
                self.Dp[p] = dNew if dNew < 1 else 1
                

            #----DAMAGE ROUTINES END----

            #P2G for velocity, force update, and update velocity
            for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                gridIdx = base + offset
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]
                
                dweight = ti.Vector.zero(float,2)
                dweight[0] = self.invDx * dw[i][0] * w[j][1]
                dweight[1] = self.invDx * w[i][0] * dw[j][1]

                force = -self.Vp[p] * kirchoff @ dweight

                if self.separable[gridIdx] == -1: 
                    #treat node as one field
                    if(self.useAPIC):
                        self.grid_v[gridIdx, 0] += self.mp[p] * weight * (self.v[p] + self.C[p] @ dpos) #momentum transfer (APIC)
                        self.grid_vn[gridIdx, 0] += self.mp[p] * weight * (self.v[p] + self.C[p] @ dpos) #momentum transfer (APIC) for saving v_i^n
                    else:
                        self.grid_v[gridIdx, 0] += self.mp[p] * weight * self.v[p] #momentum transfer (PIC)
                        self.grid_vn[gridIdx, 0] += self.mp[p] * weight * self.v[p] #momentum transfer (PIC) for saving v_i^n

                    self.grid_v[gridIdx, 0] += self.dt * force #add force to update velocity, don't divide by mass bc this is actually updating MOMENTUM

                    self.grid_n[gridIdx, 0] += dweight * self.mp[p] #add to the normal for this field at this grid node, remember we need to normalize it later!

                else:
                    #treat node as having two fields
                    fieldIdx = self.particleAF[p][i*3 + j] #grab the field that this particle is in for this node
                    
                    if(self.useAPIC):
                        self.grid_v[gridIdx, fieldIdx] += self.mp[p] * weight * (self.v[p] + self.C[p] @ dpos) #momentum transfer (APIC)
                        self.grid_vn[gridIdx, fieldIdx] += self.mp[p] * weight * (self.v[p] + self.C[p] @ dpos) #momentum transfer (APIC) for saving v_i^n
                    else:
                        self.grid_v[gridIdx, fieldIdx] += self.mp[p] * weight * self.v[p] #momentum transfer (PIC)
                        self.grid_vn[gridIdx, fieldIdx] += self.mp[p] * weight * self.v[p] #momentum transfer (PIC) for saving v_i^n
                    
                    self.grid_v[gridIdx, fieldIdx] += self.dt * force #add force to update velocity, don't divide by mass bc this is actually updating MOMENTUM
                    self.grid_n[gridIdx, fieldIdx] += dweight * self.mp[p] #add to the normal for this field at this grid node, remember we need to normalize it later!

    @ti.kernel
    def addGravity(self):
         #Add Gravity
        for i, j in self.grid_m:
            if self.separable[i,j] != -1:
                self.grid_v[i, j, 0] += self.dt * self.gravity[None] * self.grid_m[i,j][0] # gravity (single field version)
            else:
                self.grid_v[i, j, 0] += self.dt * self.gravity[None] * self.grid_m[i, j][0] # gravity, field 1
                self.grid_v[i, j, 1] += self.dt * self.gravity[None] * self.grid_m[i, j][1] # gravity, field 2

    @ti.kernel
    def computeContactForces(self):
        #Frictional Contact Forces
        for i,j in self.grid_m:
            if self.separable[i,j] != -1: #only apply these forces to nodes with two fields
                #momentium
                q_1 = self.grid_v[i, j, 0]
                q_2 = self.grid_v[i, j, 1]
                q_cm = q_1 + q_2 

                #mass
                m_1 = self.grid_m[i, j][0]
                m_2 = self.grid_m[i, j][1]
                m_cm = m_1 + m_2

                #velocity
                v_1 = self.grid_v[i, j, 0] / m_1
                v_2 = self.grid_v[i, j, 1] / m_2
                v_cm = q_cm / m_cm #NOTE: we need to compute this like this to conserve mass and momentum

                #normals
                n_1 = self.grid_n[i, j, 0].normalized() #don't forget to normalize these!!
                n_2 = self.grid_n[i, j, 1].normalized()
                n_cm1 = (n_1 - n_2).normalized()
                n_cm2 = -n_cm1

                #orthonormal basis for tengent force
                s_cm1 = ti.Vector([-1 * n_cm1[1], n_cm1[0]]) #orthogonal to n_cm
                s_cm2 = s_cm1

                #initialize to hold contact force for each field 
                f_c1 = ti.Vector([0.0, 0.0])
                f_c2 = ti.Vector([0.0, 0.0])

                #Compute these for each field regardless of separable or not
                fNormal1 =  (m_1 / self.dt) * (v_cm - v_1).dot(n_cm1)
                fNormal2 =  (m_2 / self.dt) * (v_cm - v_2).dot(n_cm2)

                fTan1 = (m_1 / self.dt) * (v_cm - v_1).dot(s_cm1)
                fTan2 = (m_2 / self.dt) * (v_cm - v_2).dot(s_cm2)

                fTanComp1 = fTan1 * s_cm1
                fTanComp2 = fTan2 * s_cm2

                fTanMag1 = fTanComp1.norm()
                fTanMag2 = fTanComp2.norm()

                fTanSign1 = 1.0 if fTanMag1 > 0 else 0.0 #L2 norm is always >= 0
                fTanSign2 = 1.0 if fTanMag2 > 0 else 0.0

                tanDirection1 = ti.Vector([0.0, 0.0]) if fTanSign1 == 0.0 else fTanComp1.normalized() #prevent nan directions
                tanDirection2 = ti.Vector([0.0, 0.0]) if fTanSign2 == 0.0 else fTanComp2.normalized()

                scale = 1e15

                #if(i * self.nGrid + j == 883):# and self.verbose):

                    
                    # print('Data for grid node', i*self.nGrid + j, ':')
                    # print('q_1:', q_1*scale)
                    # print('q_2:', q_2*scale)
                    # print('q_cm:', q_cm*scale)
                    # print()
                    # print('m_1:', m_1*scale)
                    # print('m_2:', m_2*scale)
                    # print('m_cm:', m_cm*scale)
                    # print()
                    # print('v_1:', v_1*scale)
                    # print('v_2:', v_2*scale)
                    # print('v_cm:', v_cm*scale)
                    # print()
                    # print('n_1:', n_1*scale)
                    # print('n_2:', n_2*scale)
                    # print('n_cm1:', n_cm1*scale)
                    # print('n_cm2:', n_cm2*scale)
                    # print('s_cm1:', s_cm1*scale)
                    # print('s_cm2:', s_cm2*scale)
                    # print()
                    # print('m1/dt:', m_1 / self.dt)
                    # print('m2/dt:', m_2 / self.dt)
                    # print('vcm-v1 dot n_cm1:', (v_cm - v_1).dot(n_cm1)*scale)
                    # print('vcm-v1 dot s_cm1:', (v_cm - v_1).dot(s_cm1)*scale)
                    # print('vcm-v2 dot n_cm2:', (v_cm - v_2).dot(n_cm2)*scale)
                    # print('vcm-v2 dot s_cm2:', (v_cm - v_2).dot(s_cm2)*scale)
                    # print('fNormal1:', fNormal1*scale)
                    # print('fNormal2:', fNormal2*scale)
                    # print('fTan1:', fTan1*scale)
                    # print('fTan2:', fTan2*scale)
                    # print('fTanSign1:', fTanSign1)
                    # print('fTanSign2:', fTanSign2)
                    # print('tanDirection1:', tanDirection1)
                    # print('tanDirection2:', tanDirection2)

                fieldOne = 0
                fieldTwo = 0

                if self.separable[i,j] == 1:            
                    #two fields and are separable
                    # if (v_cm - v_1).dot(n_cm1) > 0:
                    #     tanMin = self.fricCoeff * abs(fNormal1) if self.fricCoeff * abs(fNormal1) < abs(fTanMag1) else abs(fTanMag1)
                    #     f_c1 += (fNormal1 * n_cm1) + (tanMin * fTanSign1 * tanDirection1)
                    #     fieldOne = 1
                
                    # if (v_cm - v_2).dot(n_cm2) > 0:
                    #     tanMin = self.fricCoeff * abs(fNormal2) if self.fricCoeff * abs(fNormal2) < abs(fTanMag2) else abs(fTanMag2)
                    #     f_c2 += (fNormal2 * n_cm2) + (tanMin * fTanSign2 * tanDirection2)
                    #     fieldTwo = 1

                    if (v_cm - v_1).dot(n_cm1) + (v_cm - v_2).dot(n_cm2) < 0:
                        tanMin1 = self.fricCoeff * abs(fNormal1) if self.fricCoeff * abs(fNormal1) < abs(fTanMag1) else abs(fTanMag1)
                        f_c1 += (fNormal1 * n_cm1) + (tanMin1 * fTanSign1 * tanDirection1)
                        fieldOne = 1
                        tanMin1 = self.fricCoeff * abs(fNormal2) if self.fricCoeff * abs(fNormal2) < abs(fTanMag2) else abs(fTanMag2)
                        f_c2 += (fNormal2 * n_cm2) + (tanMin1 * fTanSign2 * tanDirection2)
                        fieldTwo = 1        

                    if(i * self.nGrid + j == 883 and fieldOne & fieldTwo and self.verbose):
                        print('m_1:', m_1)
                        print('m_2:', m_2)
                        print('m1/dt:', m_1 / self.dt)
                        print('m2/dt:', m_2 / self.dt)
                        print('v_1:', v_1)
                        print('v_2:', v_2)
                        print('v_cm:', v_cm)
                        print('n_cm1:', n_cm1)
                        print('n_cm2:', n_cm2)
                        print('fNormal1:', fNormal1)
                        print('fNormal2:', fNormal2)
                        print('vcm-v1 dot n_cm1:', (v_cm - v_1).dot(n_cm1))
                        print('vcm-v2 dot n_cm2:', (v_cm - v_2).dot(n_cm2))
                        v1New = v_1 + (self.dt * (-f_c2 / m_1))
                        v2New = v_2 + (self.dt * (-f_c1 / m_2))
                        print('vcm-v1New dot n_cm1:', (v_cm - v1New).dot(n_cm1)) 
                        print('vcm-v2New dot n_cm2:', (v_cm - v2New).dot(n_cm2))
                        print()


                else:
                    #two fields but not separable, treat as one field, but each gets an update
                    #NOTE: yellow node update reduces to v1 = v_cm, v2 = v_cm
                    f_c1 = v_cm #we're now updating velocity (divide out mass before adding friction force)
                    f_c2 = v_cm

                if (fieldOne ^ fieldTwo):
                    print('ERROR ERROR ERROR: fieldOne:', fieldOne, 'fieldTwo:', fieldTwo)

                #Now save these forces for later
                self.grid_f[i,j,0] = f_c1
                self.grid_f[i,j,1] = f_c2

    @ti.kernel
    def momentumToVelocityAndAddContact(self):
        #Convert Momentum to Velocity And Add Friction Forces
        for i, j in self.grid_m:    
            if self.separable[i,j] == -1:
                #treat as one field
                nodalMass = self.grid_m[i,j][0]
                if nodalMass > 0: #if there is mass at this node
                    self.grid_v[i, j, 0] = (1 / nodalMass) * self.grid_v[i, j, 0] # Momentum to velocity for v_i^n+1
                    self.grid_vn[i, j, 0] = (1 / nodalMass) * self.grid_vn[i, j, 0] # Momentum to velocity for v_i^n
                    
            else:
                #treat node as having two fields
                nodalMass1 = self.grid_m[i,j][0]
                nodalMass2 = self.grid_m[i,j][1]
                if nodalMass1 > 0 and nodalMass2 > 0: #if there is mass at this node
                    self.grid_v[i, j, 0] = (1 / nodalMass1) * self.grid_v[i, j, 0] # Momentum to velocity, field 1
                    self.grid_v[i, j, 1] = (1 / nodalMass2) * self.grid_v[i, j, 1] # Momentum to velocity, field 2
                    self.grid_vn[i, j, 0] = (1 / nodalMass1) * self.grid_vn[i, j, 0] # Momentum to velocity, field 1 for v_i^n
                    self.grid_vn[i, j, 1] = (1 / nodalMass2) * self.grid_vn[i, j, 1] # Momentum to velocity, field 2 for v_i^n

                    if(self.useFrictionalContact):
                        if self.separable[i,j] == 1:
                            self.grid_v[i,j,0] += (self.grid_f[i,j,0] / nodalMass1) * self.dt # use field 2 force to update field 1 particles (for green nodes, ie separable contact)
                            self.grid_v[i,j,1] += (self.grid_f[i,j,1] / nodalMass2) * self.dt # use field 1 force to update field 2 particles
                            #self.grid_v[i,j,0] += (-self.grid_f[i,j,1] / nodalMass1) * self.dt # use field 2 force to update field 1 particles (for green nodes, ie separable contact)
                            #self.grid_v[i,j,1] += (-self.grid_f[i,j,0] / nodalMass2) * self.dt # use field 1 force to update field 2 particles
                        else:
                            # self.grid_v[i,j,0] += self.dt * self.grid_f[i,j,0] #use field 1 for field 1 (yellow nodes)
                            # self.grid_v[i,j,1] += self.dt * self.grid_f[i,j,1] #field 2 for field 2 (yellow)
                            self.grid_v[i,j,0] = self.grid_f[i,j,0] #stored v_cm * m_1, so we just set to this!
                            self.grid_v[i,j,1] = self.grid_f[i,j,1] #stored v_cm * m_2, so we just set to this!


    @ti.kernel
    def G2P(self):
        # grid to particle (G2P)
        for p in range(self.numParticles): 
            base = (self.x[p] * self.invDx - 0.5).cast(int)
            fx = self.x[p] * self.invDx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]
            new_v_PIC = ti.Vector.zero(float, 2) #contain PIC velocity
            new_v_FLIP = ti.Vector.zero(float, 2) #contain FLIP velocity
            new_v = ti.Vector.zero(float, 2) #contain the blend velocity
            new_C = ti.Matrix.zero(float, 2, 2)
            new_F = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)): # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - fx
                gridIdx = base + ti.Vector([i, j])
                g_v_n = self.grid_vn[gridIdx, 0] #v_i^n field 1
                g_v2_n = self.grid_vn[gridIdx, 1] #v_i^n field 2
                g_v_np1 = self.grid_v[gridIdx, 0] #v_i^n+1 field 1
                g_v2_np1 = self.grid_v[gridIdx, 1] #v_i^n+1 field 2
                weight = w[i][0] * w[j][1]

                dweight = ti.Vector.zero(float,2)
                dweight[0] = self.invDx * dw[i][0] * w[j][1]
                dweight[1] = self.invDx * w[i][0] * dw[j][1]

                if self.separable[gridIdx] == -1:
                    #treat as one field
                    new_v_PIC += weight * g_v_np1
                    new_v_FLIP += weight * (g_v_np1 - g_v_n)
                    new_C += 4 * self.invDx * weight * g_v_np1.outer_product(dpos)
                    new_F += g_v_np1.outer_product(dweight)
                else:
                    #node has two fields so choose the correct velocity contribution from the node
                    fieldIdx = self.particleAF[p][i*3 + j] #grab the field that this particle is in for this node
                    if fieldIdx == 0:
                        new_v_PIC += weight * g_v_np1
                        new_v_FLIP += weight * (g_v_np1 - g_v_n)
                        new_C += 4 * self.invDx * weight * g_v_np1.outer_product(dpos)
                        new_F += g_v_np1.outer_product(dweight)
                    else:
                        new_v_PIC += weight * g_v2_np1
                        new_v_FLIP += weight * (g_v2_np1 - g_v2_n)
                        new_C += 4 * self.invDx * weight * g_v2_np1.outer_product(dpos)
                        new_F += g_v2_np1.outer_product(dweight)

            #Finish computing FLIP velocity: v_p^n+1 = v_p^n + dt (v_i^n+1 - v_i^n) * wip
            new_v_FLIP = self.v[p] + (self.dt * new_v_FLIP)

            #Compute the blend
            new_v = (self.flipPicRatio * new_v_FLIP) + ((1.0 - self.flipPicRatio) * new_v_PIC)

            self.v[p], self.C[p] = new_v, new_C #set v_p n+1 to be the blended velocity

            self.x[p] += self.dt * new_v_PIC # advection, use PIC velocity for advection regardless of PIC, FLIP, or APIC
            self.F[p] = (ti.Matrix.identity(float, 2) + (self.dt * new_F)) @ self.F[p] #updateF (explicitMPM way)

    #------------Collision Objects---------------

    def addHalfSpace(self, center, normal, surface=surfaceSticky, friction = 0.0):
        #Code adapted from mpm_solver.py here: https://github.com/taichi-dev/taichi_elements/blob/master/engine/mpm_solver.py
        center = list(center)
        # normalize normal
        normal_scale = 1.0 / math.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)

        if surface == self.surfaceSticky and friction != 0:
            raise ValueError('friction must be 0 on sticky surfaces.')

        @ti.kernel
        def collide():
            for I in ti.grouped(self.grid_m):
                if self.separable[I] == -1:
                    #treat as one field
                    nodalMass = self.grid_m[I][0]
                    if nodalMass > 0:
                        offset = I * self.dx - ti.Vector(center)
                        n = ti.Vector(normal)
                        if offset.dot(n) < 0:
                            if ti.static(surface == self.surfaceSticky):
                                self.grid_v[I,0] = ti.Vector.zero(ti.f64, self.dim) #if sticky, set velocity to all zero
                            else:
                                v = self.grid_v[I,0] #divide out the mass to get velocity
                                normal_component = n.dot(v)

                                if ti.static(surface == self.surfaceSlip):
                                    # Project out all normal component
                                    v = v - n * normal_component
                                else:
                                    # Project out only inward normal component
                                    v = v - n * min(normal_component, 0)

                                if normal_component < 0 and v.norm() > 1e-30:
                                    # apply friction here
                                    v = v.normalized() * max(0, v.norm() + normal_component * friction)

                                self.grid_v[I,0] = v

                else:
                    #treat as two fields
                    nodalMass1 = self.grid_m[I][0]
                    nodalMass2 = self.grid_m[I][1]
                    if nodalMass1 > 0 and nodalMass2 > 0:
                        offset = I * self.dx - ti.Vector(center)
                        n = ti.Vector(normal)
                        if offset.dot(n) < 0:
                            if ti.static(surface == self.surfaceSticky):
                                self.grid_v[I,0] = ti.Vector.zero(ti.f64, self.dim) #if sticky, set velocity to all zero
                                self.grid_v[I,1] = ti.Vector.zero(ti.f64, self.dim) #both fields get zero

                            else:
                                v1 = self.grid_v[I,0] #divide out the mass to get velocity
                                v2 = self.grid_v[I,1] #divide out the mass to get velocity

                                normal_component1 = n.dot(v1)
                                normal_component2 = n.dot(v2)

                                if ti.static(surface == self.surfaceSlip):
                                    # Project out all normal component
                                    v1 = v1 - n * normal_component1
                                    v2 = v2 - n * normal_component2
                                else:
                                    # Project out only inward normal component
                                    v1 = v1 - n * min(normal_component1, 0)
                                    v2 = v2 - n * min(normal_component2, 0)

                                if normal_component1 < 0 and v1.norm() > 1e-30:
                                    # apply friction here
                                    v1 = v1.normalized() * max(0, v1.norm() + normal_component1 * friction)

                                if normal_component2 < 0 and v2.norm() > 1e-30:
                                    # apply friction here
                                    v2 = v2.normalized() * max(0, v2.norm() + normal_component2 * friction)

                                self.grid_v[I,0] = v1
                                self.grid_v[I,1] = v2

        self.gridPostProcess.append(collide)

    #------------Simulation Routines---------------

    #Simulation substep
    def substep(self):

        with Timer("Reinitialize Structures"):
            self.reinitializeStructures()
        with Timer("Back Grid Sort"):
            self.backGridSort()
        with Timer("Particle Neighbor Sorting"):
            self.particleNeighborSorting()
        with Timer("Surface Detection"):
            self.surfaceDetection()
        with Timer("Compute Particle DGs"):
            self.computeParticleDG()
        with Timer("Compute Grid DGs"):
            self.computeGridDG()
        with Timer("Mass P2G"):
            self.massP2G()
        with Timer("Compute Separability"):
            self.computeSeparability()
        with Timer("Momentum P2G and Forces"):
            self.momentumP2GandForces()
        with Timer("Add Gravity"):
            self.addGravity()
        with Timer("Frictional Contact"):
            self.computeContactForces()
        with Timer("Momentum to Velocity & Add Friction"):
            self.momentumToVelocityAndAddContact()
        with Timer("Boundaries"):
            for func in self.gridPostProcess:
                func()
        with Timer("G2P"):
            self.G2P()

    @ti.kernel
    def reset(self, arr: ti.ext_arr(), partCount: ti.ext_arr(), initVel: ti.ext_arr(), pMasses: ti.ext_arr(), pVols: ti.ext_arr(), surfaceThresholds: ti.ext_arr()):
        self.gravity[None] = [0, self.gravMag]
        for i in range(self.numParticles):
            self.x[i] = [ti.cast(arr[i,0], ti.f64), ti.cast(arr[i,1], ti.f64)]
            self.material[i] = 0
            self.v[i] = [0, 0]
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.Jp[i] = 1
            self.C[i] = ti.Matrix.zero(float, 2, 2)
            self.Dp[i] = 0
            self.sp[i] = 0
            # if (self.x[i][0] > 0.495) and (self.x[i][0] < 0.503): #put damaged particles as a band in the center
            #     self.Dp[i] = 1
            #     self.material[i] = 1
        
        for serial in range(1):
            idx = 0
            for i in range(self.numObjects):
                objIdx = partCount[i]
                for j in range(objIdx):
                    self.v[idx] = [ti.cast(initVel[i,0], ti.f64), ti.cast(initVel[i,1], ti.f64)]
                    self.mp[idx] = ti.cast(pMasses[i], ti.f64)
                    self.Vp[idx] = ti.cast(pVols[i], ti.f64)
                    self.st[idx] = ti.cast(surfaceThresholds[i], ti.f64)
                    idx += 1 

    def writeData(self, frame: ti.i32, s: ti.i32):
        
        if(s == -1):
            print('[Simulation]: Writing frame ', frame, '...')
        else:
            print('[Simulation]: Writing substep ', s, 'of frame ', frame, '...')

        #Write PLY Files
        np_x = self.x.to_numpy()
        writer = ti.PLYWriter(num_vertices=self.numParticles)
        writer.add_vertex_pos(np_x[:,0], np_x[:, 1], np.zeros(self.numParticles)) #add position
        writer.add_vertex_channel("m_p", "double", self.mp.to_numpy()) #add damage
        writer.add_vertex_channel("Dp", "double", self.Dp.to_numpy()) #add damage
        writer.add_vertex_channel("sp", "int", self.sp.to_numpy()) #add surface tracking
        writer.add_vertex_channel("DGx", "double", self.particleDG.to_numpy()[:,0]) #add particle DG x
        writer.add_vertex_channel("DGy", "double", self.particleDG.to_numpy()[:,1]) #add particle DG y
        if(s == -1):
            writer.export_frame(frame, self.outputPath)
        else:
            writer.export_frame(frame * self.numSubsteps + s, self.outputPath)

        #Construct positions for grid nodes
        gridX = np.zeros((self.nGrid**2, 2), dtype=float) #format as 1d array of nodal positions
        np_separability = np.zeros(self.nGrid**2, dtype=int)
        gridNormals = np.zeros((self.nGrid**2, 4), dtype=float)
        gridMasses = np.zeros((self.nGrid**2, 2), dtype=float)
        gridVelocities = np.zeros((self.nGrid**2, 4), dtype=float)
        gridFrictionForces = np.zeros((self.nGrid**2, 4), dtype=float)
        np_DG = np.zeros((self.nGrid**2, 2), dtype=float)
        for i in range(self.nGrid):
            for j in range(self.nGrid):
                gridIdx = i * self.nGrid + j
                gridX[gridIdx,0] = i * self.dx
                gridX[gridIdx,1] = j * self.dx
                np_separability[gridIdx] = self.separable[i,j] #grab separability
                np_DG[gridIdx, 0] = self.gridDG[i,j][0]
                np_DG[gridIdx, 1] = self.gridDG[i,j][1]
                gridVelocities[gridIdx, 0] = self.grid_v[i, j, 0][0]
                gridVelocities[gridIdx, 1] = self.grid_v[i, j, 0][1]
                gridVelocities[gridIdx, 2] = self.grid_v[i, j, 1][0]
                gridVelocities[gridIdx, 3] = self.grid_v[i, j, 1][1]
                gridMasses[gridIdx, 0] = self.grid_m[i,j][0]
                gridMasses[gridIdx, 1] = self.grid_m[i,j][1]
                gridNormals[gridIdx, 0] = self.grid_n[i, j, 0][0]
                gridNormals[gridIdx, 1] = self.grid_n[i, j, 0][1]
                gridNormals[gridIdx, 2] = self.grid_n[i, j, 1][0]
                gridNormals[gridIdx, 3] = self.grid_n[i, j, 1][1]
                if self.separable[i,j] != -1:
                    gridFrictionForces[gridIdx, 0] = self.grid_f[i, j, 0][0]
                    gridFrictionForces[gridIdx, 1] = self.grid_f[i, j, 0][1]
                    gridFrictionForces[gridIdx, 2] = self.grid_f[i, j, 1][0]
                    gridFrictionForces[gridIdx, 3] = self.grid_f[i, j, 1][1]
        writer2 = ti.PLYWriter(num_vertices=self.nGrid**2)
        writer2.add_vertex_pos(gridX[:,0], gridX[:, 1], np.zeros(self.nGrid**2)) #add position
        writer2.add_vertex_channel("sep", "int", np_separability)
        writer2.add_vertex_channel("DGx", "double", np_DG[:,0]) #add particle DG x
        writer2.add_vertex_channel("DGy", "double", np_DG[:,1]) #add particle DG y
        writer2.add_vertex_channel("N_field1_x", "double", gridNormals[:,0]) #add grid_n for field 1 x
        writer2.add_vertex_channel("N_field1_y", "double", gridNormals[:,1]) #add grid_n for field 1 y
        writer2.add_vertex_channel("N_field2_x", "double", gridNormals[:,2]) #add grid_n for field 2 x
        writer2.add_vertex_channel("N_field2_y", "double", gridNormals[:,3]) #add grid_n for field 2 y
        writer2.add_vertex_channel("f_field1_x", "double", gridFrictionForces[:,0]) #add grid_f for field 1 x
        writer2.add_vertex_channel("f_field1_y", "double", gridFrictionForces[:,1]) #add grid_f for field 1 y
        writer2.add_vertex_channel("f_field2_x", "double", gridFrictionForces[:,2]) #add grid_f for field 2 x
        writer2.add_vertex_channel("f_field2_y", "double", gridFrictionForces[:,3]) #add grid_f for field 2 y
        writer2.add_vertex_channel("v_field1_x", "double", gridVelocities[:,0])
        writer2.add_vertex_channel("v_field1_y", "double", gridVelocities[:,1])
        writer2.add_vertex_channel("v_field2_x", "double", gridVelocities[:,2])
        writer2.add_vertex_channel("v_field2_y", "double", gridVelocities[:,3])
        writer2.add_vertex_channel("m1", "double", gridMasses[:,0])
        writer2.add_vertex_channel("m2", "double", gridMasses[:,1])

        if(s == -1):
            writer2.export_frame(frame, self.outputPath2)
        else:
            writer2.export_frame(frame * self.numSubsteps + s, self.outputPath2)

    def simulate(self):
        print("[Simulation] Particle Count: ", self.numParticles)
        print("[Simulation] Grid Dx: ", self.dx)
        print("[Simulation] Time Step: ", self.dt)
        self.reset(self.vertices, self.particleCounts, self.initialVelocity, self.pMasses, self.pVolumes, self.surfaceThresholds) #init
        for frame in range(self.endFrame):
            with Timer("Compute Frame"):
                if(self.verbose == False): 
                    with Timer("Visualization"):
                        self.writeData(frame, -1) #NOTE: activate to write frames only
                for s in range(self.numSubsteps):
                    if(self.verbose): 
                        with Timer("Visualization"):
                            self.writeData(frame, s) #NOTE: activate to write every substep
                    with Timer("Compute Substep"):
                        self.substep()
        Timer_Print()