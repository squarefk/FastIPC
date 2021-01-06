import taichi as ti
import numpy as np
import math
from common.utils.timer import *
from projects.mpm.basic.fixed_corotated import *
from projects.mpm.basic.sparse_matrix import SparseMatrix, CGSolver
#import scipy.sparse.linalg

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
    def __init__(self, endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThreshold, useDFG = True, frictionCoefficient = 0.0, verbose = False, useAPIC = False, flipPicRatio = 0.0, symplectic = True):
        
        #Simulation Parameters
        self.endFrame = endFrame
        self.fps = fps
        self.frameDt = 1.0 / fps
        self.dt = dt
        self.maxDt = self.frameDt if self.frameDt < dt else dt #used to cap implicit time stepping
        self.minDt = 1e-6
        self.elapsedTime = 0.0 #keep a concept of elapsed time
        self.numSubsteps = int(self.frameDt // dt)
        self.dim = len(vertices[0])
        if self.dim != 2 and self.dim != 3:
            raise ValueError('ERROR: Dimension must be 2 or 3')
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
        self.st = surfaceThreshold
        self.EList = np.array(EList)
        self.nuList = np.array(nuList)
        self.dx = dx
        self.invDx = 1.0 / dx
        self.nGrid = ti.ceil(self.invDx)
        self.gravMag = gravity
        self.useFrictionalContact = True #not sure there's any reason not to use this at this point!
        self.useDFG = useDFG #determine whether we should be using partitioning or not (turn off to use explicit MPM)
        self.verbose = verbose
        self.useAPIC = useAPIC
        self.flipPicRatio = flipPicRatio #default to 0 which means full PIC
        if flipPicRatio < 0.0 or flipPicRatio > 1.0:
            raise ValueError('flipPicRatio must be between 0 and 1')
        self.useDamage = False
        self.activeNodes = ti.field(dtype=int, shape=()) #track current active nodes
        self.separableNodes = ti.field(dtype=int, shape=()) #track current separable nodes
        self.symplectic = symplectic
        self.gravity = ti.Vector.field(self.dim, dtype=float, shape=())
        
        #Collision Variables
        self.collisionCallbacks = [] #hold function callbacks for post processing velocity
        self.transformCallbacks = [] #hold callbacks for moving our boundaries
        #self.collisionTranslations = [] #hold the translations for these moving boundaries
        self.collisionObjectCount = 0
        self.collisionObjectCenters = ti.Vector.field(self.dim, dtype=float, shape=16) #allow up to 16 collision objects for now
        self.collisionVelocities = ti.Vector.field(self.dim, dtype=float, shape=16) #hold the translations for these moving boundaries, we'll also use these to set vi for sticky bounds
        self.collisionTypes = ti.field(dtype=int, shape=16) #store collision types

        #External Forces
        self.useImpulse = False
        self.impulseCenter = (0.0, 0.0) if self.dim == 2 else (0.0, 0.0, 0.0)
        self.impulseStrength = 0.0
        self.impulseStartTime = 0.0
        self.impulseDuration = 0.0

        #AnisoMPM Damage Parameters
        self.eta = 1.0
        self.sigmaCRef = 1.0
        self.zeta = 1.0
        self.useAnisoMPMDamage = False

        #Rankine Damage Parameters
        self.percentStretch =  -1.0
        self.l0 = 0.5 * dx #as usual, close to the sqrt(2) * dx that they use
        self.Gf = 1.0
        self.useRankineDamage = False
        
        #Weibull Params
        self.useWeibull = False
        self.sigmaFRef = -1.0
        self.m = 1.0
        self.vRef = 1.0

        #Time to Failure Damage Parameters, many of these will be set later when we add the damage model
        self.damageList = np.array(EList) #dummy list
        self.cf = 1.0
        self.timeToFail = 1.0

        #DFG Parameters
        self.rp = (3*(dx**2))**0.5 if self.dim == 3 else (2*(dx**2))**0.5 #set rp based on dx (this changes if dx != dy)
        self.dMin = 0.25
        self.minDp = 1.0
        self.fricCoeff = frictionCoefficient

        #Barrier Function Parameters
        self.chat = 0.001
        self.constraintsViolated = ti.field(dtype=int, shape=())

        #Particle Structures
        #---Lame Parameters
        self.mu = ti.field(dtype=float)
        self.la = ti.field(dtype=float)
        #---General Simulation
        self.x = ti.Vector.field(self.dim, dtype=float) # position
        self.v = ti.Vector.field(self.dim, dtype=float) # velocity
        self.C = ti.Matrix.field(self.dim, self.dim, dtype=float) # affine velocity field
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=float) # deformation gradient
        self.F_backup = ti.Matrix.field(self.dim, self.dim, dtype=float) # deformation gradient
        self.material = ti.field(dtype=int) # material id
        self.mp = ti.field(dtype=float) # particle masses
        self.Vp = ti.field(dtype=float) # particle volumes
        self.Jp = ti.field(dtype=float) # plastic deformation
        #---DFG Fields
        self.Dp = ti.field(dtype=float) #particle damage
        self.sp = ti.field(dtype=int) #particle surface boolean int -- 1 = surface particle, 0 = interior particle
        self.particleDG = ti.Vector.field(self.dim, dtype=float) #keep track of particle damage gradients 
        #self.st = ti.field(dtype=float) #now we can have different thresholds for different objects and particle distributions!        
        #---AnisoMPM Damage
        self.sigmaC = ti.field(dtype=float) #each particle can have different sigmaC
        self.dTildeH = ti.field(dtype=float) #keep track of the maximum driving force seen by this particle
        self.damageLaplacians = ti.field(dtype=float) #store the damage laplacians
        self.useAnisoMPMDamageList = ti.field(dtype=int)
        #---Rankine Damage
        self.Hs = ti.field(dtype=float) #weibull will make these different from eahc other
        self.sigmaF = ti.field(dtype=float) #each particle can have different sigmaF based on Weibull dist
        self.sigmaMax = ti.field(dtype=float) #track sigmaMax for each particle to visualize stress
        self.useRankineDamageList = ti.field(dtype=int)
        #---Time To Failure Damage
        self.useTimeToFailureDamageList = ti.field(dtype=int) 
        #---Active Fields Tracking
        self.particleAF = ti.field(dtype=int) #store which activefield each particle belongs to for the 9 grid nodes it maps to
        #---Caching Weights and Grid Indeces
        self.p_cached_w = ti.field(dtype=float)
        self.p_cached_dw = ti.Vector.field(self.dim, dtype=float)
        self.p_cached_idx = ti.field(dtype=int)
        #---Neighbor Search Particle Structures
        self.particleNumNeighbors = ti.field(dtype=int)  #track how many neighbors each particle has
        self.particleNeighbors = ti.field(dtype=int)     #map a particle to its list of neighbors, particleNumNeighbors is nParticles x 1

        #Sparse Grid Structures
        #---Params
        self.grid_size = 4096
        self.grid_block_size = 128
        self.leaf_block_size = 16 if self.dim == 2 else 8
        self.indices = ti.ij if self.dim == 2 else ti.ijk
        #self.offset = tuple(-self.grid_size // 2 for _ in range(self.dim))
        self.offset = tuple(0 for _ in range(self.dim)) #NOTE: this means we assume everything to be in quadrant 1 --> this is useful for aligning with the spatial hash for neighbor search
        #---Grid Shapes for PID
        self.grid = ti.root.pointer(self.indices, self.grid_size // self.grid_block_size) # 32
        self.block = self.grid.pointer(self.indices, self.grid_block_size // self.leaf_block_size) # 8
        self.pid = ti.field(int)
        self.block.dynamic(ti.indices(self.dim), 1024 * 1024, chunk_size=self.leaf_block_size**self.dim * 8).place(self.pid, offset=self.offset + (0, ))
        #---Grid Shapes for Rest of Grid Structures
        self.grid2 = ti.root.pointer(self.indices, self.grid_size // self.grid_block_size) # 32
        self.block2 = self.grid2.pointer(self.indices, self.grid_block_size // self.leaf_block_size) # 8
        def block_component(c):
            self.block2.dense(self.indices, self.leaf_block_size).place(c, offset=self.offset) # 16 in 3D, 8 in 2D (-2048, 2048) or (0, 4096) w/o offset
        #---Grid Simulation Structures
        self.grid_d = ti.Vector.field(4, dtype=float) #always vec4, field 1 numerator, field 2 numerator, field 1 denom, field 2 denom
        self.grid_q1 = ti.Vector.field(self.dim, dtype=float)
        self.grid_q2 = ti.Vector.field(self.dim, dtype=float)
        self.grid_v1 = ti.Vector.field(self.dim, dtype=float)
        self.grid_v2 = ti.Vector.field(self.dim, dtype=float)
        self.grid_vn1 = ti.Vector.field(self.dim, dtype=float)
        self.grid_vn2 = ti.Vector.field(self.dim, dtype=float)
        self.grid_m1 = ti.field(dtype=float)
        self.grid_m2 = ti.field(dtype=float)
        self.grid_n1 = ti.Vector.field(self.dim, dtype=float)
        self.grid_n2 = ti.Vector.field(self.dim, dtype=float)
        self.grid_f1 = ti.Vector.field(self.dim, dtype=float) #Internal elasticity forces
        self.grid_f2 = ti.Vector.field(self.dim, dtype=float)
        self.grid_fct1 = ti.Vector.field(self.dim, dtype=float) #Contact forces
        self.grid_fct2 = ti.Vector.field(self.dim, dtype=float)
        self.grid_fi1 = ti.Vector.field(self.dim, dtype=float) #Impulse forces (need to store them if we do implicit so we can incorporate them into rhs)
        self.grid_fi2 = ti.Vector.field(self.dim, dtype=float)
        #---DOF Tracking
        self.grid_idx = ti.field(dtype=int)
        self.grid_sep_idx = ti.field(dtype=int)
        #---DFG Grid Structures
        self.gridDG = ti.Vector.field(self.dim, dtype=float)
        self.gridMaxNorm = ti.field(dtype=float)
        self.gridSeparability = ti.Vector.field(4, dtype=float) #numerator field 1, numerator field 2, denom field 1, denom field 2
        self.gridMaxDamage = ti.Vector.field(2, dtype=float) #max damage field 1, max damage field 2
        self.separable = ti.field(dtype=int)
        #---Neighbor Search Structures
        self.gridNumParticles = ti.field(dtype=int)      #track number of particles in each cell using cell index
        #---Barrier Function Structures
        self.gridViYi1 = ti.field(dtype=float)
        self.gridViYi2 = ti.field(dtype=float)
        self.gridCi = ti.field(dtype=float)
        
        #---Place Grid Structures
        for d in self.grid_d.entries: #grid damage structure, need both fields as well as numerator accumulator and denom accumulator
            block_component(d) 
        for q1 in self.grid_q1.entries: # grid node momentum, field 1
            block_component(q1)
        for q2 in self.grid_q2.entries: # grid node momentum, field 2
            block_component(q2)
        for v1 in self.grid_v1.entries: # grid node velocity, field 1
            block_component(v1)
        for v2 in self.grid_v2.entries: # grid node velocity, field 2
            block_component(v2)
        for vn1 in self.grid_vn1.entries: # use this to store grid v_i^n so we can use this for FLIP, field 1
            block_component(vn1)
        for vn2 in self.grid_vn2.entries: # use this to store grid v_i^n so we can use this for FLIP, field 2
            block_component(vn2)
        block_component(self.grid_m1) # grid node field 1 mass is nGrid x nGrid, each grid node has a mass for each field
        block_component(self.grid_m2) # grid node field 2 mass is nGrid x nGrid, each grid node has a mass for each field        
        for n1 in self.grid_n1.entries: # grid node normals for two field nodes, field 1
            block_component(n1)
        for n2 in self.grid_n2.entries: # grid node normals for two field nodes, field 2
            block_component(n2)
        for f1 in self.grid_f1.entries: # grid node forces so we can apply them later, field 1
            block_component(f1)
        for f2 in self.grid_f2.entries: # grid node forces so we can apply them later, field 2
            block_component(f2)
        for fct1 in self.grid_fct1.entries: # grid node contact forces for two field nodes, field 1
            block_component(fct1)
        for fct2 in self.grid_fct2.entries: # grid node contact forces for two field nodes, field 2
            block_component(fct2)
        for fi1 in self.grid_fi1.entries: # grid node impulse forces, field 1
            block_component(fi1)
        for fi2 in self.grid_fi2.entries: # grid node impulse forces, field 2
            block_component(fi2)
        block_component(self.grid_idx) #hold a mapping from active grid DOF indeces -> the DOF index
        block_component(self.grid_sep_idx) #hold a mapping from separable grid DOF indeces -> the sep DOF index
        for dg in self.gridDG.entries: #grid node damage gradients
            block_component(dg)
        block_component(self.gridMaxNorm)  #grid max norm holds the maximum DG norm found for this grid node, this will later help to determine the grid DG
        for sep in self.gridSeparability.entries: # grid separability is nGrid x nGrid x 4, each grid node has a seperability condition for each field and we need to add up the numerator and denominator
            block_component(sep)
        for maxD in self.gridMaxDamage.entries: # grid max damage is nGrid x nGrid x 2, each grid node has a max damage from each field
            block_component(maxD)
        block_component(self.separable) # whether grid node is separable or not
        block_component(self.gridNumParticles) #keep track of how many particles are at each cell of backGrid
        block_component(self.gridViYi1)
        block_component(self.gridViYi2)
        block_component(self.gridCi)

        #Neighbor Search Structures
        #---Parameters---NOTE: if these values are too low, we get data races!!! Try to keep these as high as possible (RIP to ur RAM)
        self.maxNeighbors = 2**10
        self.maxPPC = 2**10
        #---Structures---
        self.backGrid = ti.field(int)              #background grid to map grid cells to a list of particles they contain
        #Shape the neighbor fields
        backGridIndeces = ti.ijk if self.dim == 2 else ti.ijkl
        backGridShape = (self.nGrid, self.nGrid, self.maxPPC) if self.dim == 2 else (self.nGrid, self.nGrid, self.nGrid, self.maxPPC)
        ti.root.dense(backGridIndeces, backGridShape).place(self.backGrid)      #backGrid is nGrid x nGrid x maxPPC
        #gridShape2 = (self.nGrid, self.nGrid) if self.dim == 2 else (self.nGrid, self.nGrid, self.nGrid)
        #ti.root.dense(self.indices, gridShape2).place(self.gridNumParticles) 
        #self.gridShape.dense(ti.k, self.maxPPC).place(self.backGrid) 
        #NOTE: backgrid uses nGrid x nGrid even though dx != rp, but rp is ALWAYs larger than dx so it's okay!
        
        #Shape and Place Particle Structures
        #---Params
        self.max_num_particles = 2**20 #~1 million
        #---numParticles x 1 (Particle Structures)
        #self.particle = ti.root.dynamic(ti.i, self.max_num_particles, 2**19) #2**20 causes problems in CUDA (maybe asking for too much space)
        self.particle = ti.root.dense(ti.i, self.numParticles)
        self.particle.place(self.mu, self.la, self.x, self.v, self.C, self.F, self.F_backup, self.material, self.mp, self.Vp, self.Jp, self.Dp, self.sp, self.particleDG, self.sigmaC, self.dTildeH, self.damageLaplacians, self.useAnisoMPMDamageList, self.Hs, self.sigmaF, self.sigmaMax, self.useRankineDamageList, self.useTimeToFailureDamageList, self.particleNumNeighbors)
        #---numParticles x 3^dim (Stencil Structures)
        self.particle2grid = self.particle.dense(ti.j, 3**self.dim)
        self.particle2grid.place(self.particleAF, self.p_cached_w, self.p_cached_dw, self.p_cached_idx)
        #---numParticles x maxNeighbors (Neighbor Structure)
        ti.root.dense(ti.i, self.numParticles).dense(ti.j, self.maxNeighbors).place(self.particleNeighbors)
        #self.particle.dense(ti.j, self.maxNeighbors).place(self.particleNeighbors) #particleNeighbors is nParticles x maxNeighbors #this line throws an error for some reason, so use the one above instead

        #Sparse Matrix Fields
        MAX_LINEAR = 5000000
        self.data_x = ti.field(float, shape=MAX_LINEAR)
        self.entryCol = ti.field(ti.i32, shape=MAX_LINEAR)
        self.entryVal = ti.Matrix.field(self.dim, self.dim, float, shape=MAX_LINEAR)
        self.dof2idx = ti.Vector.field(self.dim, int, shape=MAX_LINEAR)
        self.sepDof2idx = ti.Vector.field(self.dim, int, shape=MAX_LINEAR)

        #Newton Optimization Variables
        self.dv = ti.field(float, shape=MAX_LINEAR)
        self.ddv = ti.field(float, shape=MAX_LINEAR)
        self.DV = ti.field(float, shape=MAX_LINEAR)
        self.rhs = ti.field(float, shape=MAX_LINEAR)
        self.boundary = ti.field(int, shape=MAX_LINEAR)
        self.matrix = SparseMatrix()
        self.cgsolver = CGSolver()

    ##########

    #General Sim Functions

    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    @ti.kernel
    def build_pid(self):
        ti.block_dim(64) #this sets the number of threads per block / block dimension
        for p in self.x:
            base = int(ti.floor(self.x[p] * self.invDx - 0.5))
            ti.append(self.pid.parent(), base - ti.Vector(list(self.offset)), p)

    #Function to compute offset -> flattened index for this grid node (e.g. offset = (0, 2, 1) --> idx = 7)
    @ti.func
    def offset2idx(self, offset):
        idx = -1
        if ti.static(self.dim == 2):
            idx = offset[0]*3+offset[1]
        if ti.static(self.dim == 3):
            idx = offset[0]*9+offset[1]*3+offset[2]
        return idx

    @ti.func
    def idx2offset(self, idx):
        offset = ti.Vector.zero(int, self.dim)
        if ti.static(self.dim == 2):
            offset[0] = idx//3
            offset[1] = idx%3
        if ti.static(self.dim == 3):
            offset[0] = idx//9
            offset[1] = (idx%9)//3
            offset[2] = (idx%9)%3
        return offset

    @ti.func
    def linear_offset(self, offset):
        return (offset[0]+2)*5+(offset[1]+2)
        # return (offset[0]+2)*25+(offset[1]+2)*5+(offset[2]+2)

    @ti.func
    def linear_offset3D(self, offset):
        return (offset[0]+2)*25+(offset[1]+2)*5+(offset[2]+2)

    ##########

    #Constitutive Model
    @ti.func 
    def kirchoff_FCR(self, F, R, J, mu, la):
        #compute Kirchoff stress using FCR elasticity
        return 2 * mu * (F - R) @ F.transpose() + ti.Matrix.identity(float, self.dim) * la * J * (J - 1) #compute kirchoff stress for FCR model (remember tau = P F^T)

    @ti.func
    def kirchoff_NeoHookean(self, F, J, mu, la):
        #compute Kirchoff stress using compressive NeoHookean elasticity (Eq 22. in Homel2016 but Kirchoff stress)
        return J * ((((la * (ti.log(J) / J)) - (mu / J)) * ti.Matrix.identity(float, self.dim)) + ((mu / J) * F @ F.transpose()))

    ##########

    #Math Tools
    @ti.func
    def eigenDecomposition2D(self, M):
        e = ti.Vector([0.0, 0.0])
        v1 = ti.Vector([0.0, 0.0])
        v2 = ti.Vector([0.0, 0.0])
        
        x11 = M[0,0]
        x12 = M[0,1]
        x21 = M[1,0]
        x22 = M[1,1]

        if x11 != 0.0 or x12 != 0.0 or x21 != 0.0 or x22 != 0.0: #only go ahead with the computation if M is not all zero
            a = 0.5 * (x11 + x22)
            b = 0.5 * (x11 - x22)
            c = x21

            c_squared = c*c
            m = (b*b + c_squared)**0.5
            k = (x11 * x22) - c_squared

            if a >= 0.0:
                e[0] = a + m
                e[1] = k / e[0] if e[0] != 0.0 else 0.0
            else:
                e[1] = a - m
                e[0] = k / e[1] if e[1] != 0.0 else 0.0

            #exhange sort
            if e[1] > e[0]:
                temp = e[0]
                e[0] = e[1]
                e[1] = temp

            v1 = ti.Vector([m+b, c]).normalized() if b >= 0 else ti.Vector([-c, b-m]).normalized()
            v2 = ti.Vector([-v1[1], v1[0]])

        return e, v1, v2

    @ti.kernel
    def testEigenDecomp(self):
        A = ti.Matrix([[0.0, 0.0],[0.0, 0.0]])
        self.eigenDecomposition2D(A)
        for i in range(10):
            base = -5.0
            interval = 10.0
            a = base + (interval*ti.random())
            b = base + (interval*ti.random())
            c = base + (interval*ti.random())
            A = ti.Matrix([[a,b],[b,c]])
            self.eigenDecomposition2D(A)

    ##########

    #Neighbor Search Routines
    @ti.func
    def backGridIdx(self, x):
        #compute int vector of backGrid indeces (recall the grid here )
        return int(x/self.rp)

    @ti.func
    def isInGrid(self, x): #TODO:3D
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

    #Barrier Energy Computations

    #NOTE: Yi is not included in these because we transferred it to the grid directly with Vi
    @ti.func
    def computeB(self, ci, chat):
        c = ci / chat
        return -(c - 1)**2 * ti.log(c) if ci < chat else 0.0

    @ti.func
    def computeBPrime(self, ci, chat):
        c = ci / chat
        return -((2.0 * (c - 1.0) * ti.log(c) / chat) + ((c - 1.0)**2 / ci)) if ci < chat else 0.0

    @ti.func
    def computeBDoublePrime(self, ci, chat):
        c = ci / chat
        return -(((2.0 * ti.log(c) + 3.0) / chat**2) - (2.0 / (ci * chat)) - (1 / ci**2)) if ci < chat else 0.0
    
    ##########

    #AnisoMPM Function Evaluations

    @ti.func
    def macaulay(self, x):
        return (x + abs(x)) / 2.0

    ##########

    #Simulation Routines
    @ti.kernel
    def reinitializeStructures(self):
        #Clear neighbor look up structures
        for p in range(self.numParticles):
            self.particleNumNeighbors[p] = 0
            for i in range(3**self.dim):
                self.particleAF[p, i] = -1
    
    @ti.kernel
    def backGridSort(self):
        #Sort particles into backGrid
        ti.block_dim(256)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            cell = self.backGridIdx(self.x[p]) #grab cell idx (vector of ints)
            offs = ti.atomic_add(self.gridNumParticles[cell], 1) #atomically add one to our grid cell's particle count NOTE: returns the OLD value before add
            #print("cell:", cell, "offs:", offs)
            #print("backGrid shape:", self.backGrid.shape)
            self.backGrid[cell, offs] = p #place particle idx into the grid cell bucket at the correct place in the cell's neighbor list (using offs)
            

    @ti.kernel
    def particleNeighborSorting(self):
        #Sort into particle neighbor lists
        #See https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py for reference
        ti.block_dim(256)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p_i = self.pid[I]
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
        ti.block_dim(256)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            pos = self.x[p]
            S = 0.0
            for i in range(self.particleNumNeighbors[p]):
                xp_i = self.particleNeighbors[p, i] #index of curr neighbor
                xp = self.x[xp_i] #grab neighbor position
                rBar = self.computeRBar(pos, xp)
                S += self.computeOmega(rBar)
            if(S <= self.st):
                self.sp[p] = 1
            elif(self.sp[p] != 1):
                self.sp[p] = 0

    @ti.kernel
    def updateDamage(self):
        ti.block_dim(256)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            
            U, sig, V = ti.svd(self.F[p])
            J = 1.0

            for d in ti.static(range(self.dim)):
                J *= sig[d, d]

            #-----TIME TO FAILURE DAMAGE-----
            if(self.useTimeToFailureDamageList[p] and self.useDFG):
                kirchoff = self.kirchoff_FCR(self.F[p], U@V.transpose(), J, self.mu[p], self.la[p])
                e, v1, v2 = self.eigenDecomposition2D(kirchoff / J) #use my eigendecomposition, comes out as three 2D vectors
                
                maxEigVal = e[0] if e[0] > e[1] else e[1] #e[0] is enforced to be larger though... so this is prob unnecessary
                
                #Update Particle Damage (only if maxEigVal is greater than this particle's sigmaF)
                if(maxEigVal > self.sigmaF[p]): 
                    dNew = self.Dp[p] + (self.dt / self.timeToFail)
                    self.Dp[p] = dNew if dNew < 1.0 else 1.0

                    #Reconstruct tensile scaled Cauchy stress
                    for d in ti.static(range(2)):
                        if e[d] > 0: e[d] *= (1 - dNew) #scale tensile eigenvalues using the new damage

                    #kirchoff = J * (e[0] * v1.outer_product(v1) + e[1] * v2.outer_product(v2))
            
            #---------RANKINE DAMAGE---------
            if self.useRankineDamageList[p] and self.useDFG:
                #Get cauchy stress and its eigenvalues and eigenvectors
                kirchoff = self.kirchoff_FCR(self.F[p], U@V.transpose(), J, self.mu[p], self.la[p]) #FCR elasticity               
                e, v1, v2 = self.eigenDecomposition2D(kirchoff / J) #use my eigendecomposition, comes out as three 2D vectors
                
                maxEigVal = e[0] if e[0] > e[1] else e[1] #e[0] is enforced to be larger though... so this is prob unnecessary
                
                #Update Particle Damage (only if maxEigVal is greater than this particle's sigmaF)
                if(maxEigVal > self.sigmaF[p]): 
                    dNew = min(1.0, (1 + self.Hs[p]) * (1 - (self.sigmaF[p] / maxEigVal))) #take min with 1 to ensure we do not exceed 1
                    self.Dp[p] = max(self.Dp[p], dNew) #irreversibility condition, cracks cannot heal 
            
    @ti.kernel
    def computeParticleDG(self):
        #Compute DG for all particles and for all grid nodes 
        # NOTE: grid node DG is based on max of mapped particle DGs, in this loop we simply create a list of candidates, then we will take max after
        ti.block_dim(256)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
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
            
            self.particleDG[p] = nablaDBar if DandS[1] != 0 else ti.Vector.zero(float, self.dim) #if particle has no neighbors set DG to zeros

            #Now iterate over the grid nodes particle p maps to to set Di!
            base = ti.floor(pos * self.invDx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            for offset in ti.static(ti.grouped(self.stencil_range())): # Loop over grid node stencil
                gridIdx = base + offset
                ti.atomic_max(self.gridMaxNorm[gridIdx], nablaDBar.norm()) #take max between our stored gridMaxNorm at this node and the norm of our nablaDBar

    @ti.kernel
    def computeGridDG(self):
        #Now iterate over particles and grid nodes again to capture the gridDGs!
        ti.block_dim(256)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            pos = self.x[p]

            base = ti.floor(pos * self.invDx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            currParticleDG = self.particleDG[p]
            for offset in ti.static(ti.grouped(self.stencil_range())): # Loop over grid node stencil
                gridIdx = base + offset
                if self.gridMaxNorm[gridIdx] == currParticleDG.norm():
                    self.gridDG[gridIdx] = currParticleDG

    @ti.kernel
    def massP2G(self):
        # P2G for mass, set active fields, and compute separability conditions
        #ti.no_activate(self.particle)
        ti.block_dim(256)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            
            #for particle p, compute base index
            base = ti.floor(self.x[p] * self.invDx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1) 
            fx = self.x[p] * self.invDx - base.cast(float)
            
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            #P2G for mass, set active fields, and compute separability conditions
            for offset in ti.static(ti.grouped(self.stencil_range())): # Loop over grid node stencil
                gridIdx = base + offset
                dpos = (offset.cast(float) - fx) * self.dx
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                
                #Cache wip and nabla wip for this particle and each of its 3^d grid nodes in the interpolation stencil
                oidx = self.offset2idx(offset)
                self.p_cached_w[p, oidx] = weight

                maxD = max(self.Dp[p], self.sp[p]) #use max of damage and surface particle markers so we detect green case correctly

                #Set Active Fields for each grid node! 
                oidx = self.offset2idx(offset)
                if self.particleDG[p].dot(self.gridDG[gridIdx]) >= 0:
                    self.grid_m1[gridIdx] += weight * self.mp[p] #add mass to active field for this particle
                    self.gridSeparability[gridIdx][0] += weight * maxD * self.mp[p] #numerator, field 1
                    self.gridSeparability[gridIdx][2] += weight * self.mp[p] #denom, field 1
                    self.particleAF[p, oidx] = 0 #set this particle's AF to 0 for this grid node
                    ti.atomic_max(self.gridMaxDamage[gridIdx][0], maxD) #compute the max damage seen in this field at this grid node
                else:
                    self.grid_m2[gridIdx] += weight * self.mp[p] #add mass to active field for this particle
                    self.gridSeparability[gridIdx][1] += weight * maxD * self.mp[p] #numerator, field 2
                    self.gridSeparability[gridIdx][3] += weight * self.mp[p] #denom, field 2
                    self.particleAF[p, oidx] = 1 #set this particle's AF to 1 for this grid node
                    ti.atomic_max(self.gridMaxDamage[gridIdx][1], maxD) #compute the max damage seen in this field at this grid node
                
                #if p == 0: print("particle0, particleAF[p, oidx]:", self.particleAF[p, oidx])

    @ti.kernel
    def computeSeparability(self):
        self.separableNodes[None] = 0 #track how many separable nodes we have

        #Iterate grid nodes to compute separability condition and maxDamage (both for each field)
        for I in ti.grouped(self.grid_m1):
            #Compute seperability for field 1 and store as idx 0
            if(self.gridSeparability[I][2] > 0): 
                self.gridSeparability[I][0] /= self.gridSeparability[I][2] #divide numerator by denominator
            else:
                self.gridSeparability[I][0] = 0.0

            #Compute seperability for field 2 and store as idx 1
            if(self.gridSeparability[I][3] > 0): 
                self.gridSeparability[I][1] /= self.gridSeparability[I][3] #divide numerator by denominator
            else:
                self.gridSeparability[I][1] = 0.0

            #NOTE: separable[I] = -1 for one field, 0 for two non-separable fields, and 1 for two separable fields
            if self.grid_m1[I] > 0 and self.grid_m2[I] > 0:
                minSep = self.gridSeparability[I][0] if self.gridSeparability[I][0] < self.gridSeparability[I][1] else self.gridSeparability[I][1]
                maxMax = self.gridMaxDamage[I][0] if self.gridMaxDamage[I][0] > self.gridMaxDamage[I][1] else self.gridMaxDamage[I][1]
                if maxMax >= self.minDp and minSep > self.dMin:
                    self.separable[I] = 1

                    #Track these separable DOFs
                    idx = self.separableNodes[None].atomic_add(1)
                    self.grid_sep_idx[I] = idx
                    self.sepDof2idx[idx] = I
                else:
                    #Now add the masses together into the first field because we'll treat this as a single field
                    self.grid_m1[I] += self.grid_m2[I]
                    self.grid_m2[I] = 0.0 #empty this in case we check mass again later
                    self.separable[I] = 0

    @ti.kernel
    def damageP2G(self):

        #damageP2G so we can compute the Laplacians
        ti.block_dim(256)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I] 
            
            #for particle p, compute base index
            base = ti.floor(self.x[p] * self.invDx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)          
            fx = self.x[p] * self.invDx - base.cast(float)
            
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2] #2x3 in 2d

            #Add damage contributions to grid nodes
            for offset in ti.static(ti.grouped(self.stencil_range())): # Loop over grid node stencil
                gridIdx = base + offset   
                oidx = self.offset2idx(offset)
                weight = self.p_cached_w[p, oidx]

                #Treat as either single-field or two-field
                if self.separable[gridIdx] != 1:
                    #Single Field
                    self.grid_d[gridIdx][0] += weight * self.Dp[p]
                    self.grid_d[gridIdx][2] += weight
                else:
                    #Two-Field
                    oidx = self.offset2idx(offset)
                    fieldIdx = self.particleAF[p, oidx]
                    if fieldIdx == 0:
                        self.grid_d[gridIdx][0] += weight * self.Dp[p]
                        self.grid_d[gridIdx][2] += weight
                    else:
                        self.grid_d[gridIdx][1] += weight * self.Dp[p]
                        self.grid_d[gridIdx][3] += weight
        
        #Now divide out the denominators for grid damage     
        for I in ti.grouped(self.grid_m1):
            if(self.grid_d[I][2] > 0): 
                self.grid_d[I][0] /= self.grid_d[I][2] #divide numerator by denominator
            else:
                self.grid_d[I][0] = 0.0
            if self.separable[I] == 1:
                if(self.grid_d[I][3] > 0): 
                    self.grid_d[I][1] /= self.grid_d[I][3] #divide numerator by denominator
                else:
                    self.grid_d[I][1] = 0.0

    @ti.kernel
    def computeLaplacians(self):
        #basically G2P but to get damage laplacians
        ti.block_dim(256)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I] 
            
            #for particle p, compute base index
            base = ti.floor(self.x[p] * self.invDx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)
            fx = self.x[p] * self.invDx - base.cast(float)
            
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2] #2x3 in 2d
            ddw = [1, -2, 1] #constant so we don't need more than 1x3 for 2d and 3d 

            #Add damage contributions to grid nodes
            self.damageLaplacians[p] = 0.0
            for offset in ti.static(ti.grouped(self.stencil_range())): # Loop over grid node stencil
                gridIdx = base + offset                

                #from ziran 2d: self.invDx**2 * (ddw[0](i) * w[1][j]) + (w[0][i] * ddw[1](j))
                #from ziran 3d: self.invDx**3 * ((ddw[0](i) * w[1][j] * w[2][k]) + (ddw[1](i) * w[0][i] * w[2][k]) + (ddw[2](i) * w[0][i] *  w[1][j]))
                laplacian = 0.0
                if ti.static(self.dim == 2):
                    laplacian = self.invDx**2 * ((ddw[offset[0]] * w[offset[1]][1]) + (w[offset[0]][0] * ddw[offset[1]]))
                else:
                    laplacian = self.invDx**3 * ((ddw[offset[0]] * w[offset[1]][1] * w[offset[2]][2]) + (ddw[offset[1]] * w[offset[0]][0] * w[offset[2]][2]) + (ddw[offset[2]] * w[offset[0]][0] *  w[offset[1]][1]))

                if self.separable[gridIdx] != 1:
                    #treat as one field
                    self.damageLaplacians[p] += self.grid_d[gridIdx][0] * laplacian
                else:
                    #node has two fields so choose the correct velocity contribution from the node
                    oidx = self.offset2idx(offset)
                    fieldIdx = self.particleAF[p, oidx] #grab the field that this particle is in for this node
                    if fieldIdx == 0:
                        self.damageLaplacians[p] += self.grid_d[gridIdx][0] * laplacian
                    else:
                        self.damageLaplacians[p] += self.grid_d[gridIdx][1] * laplacian

    @ti.kernel
    def updateAnisoMPMDamage(self):
        ti.block_dim(256)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            
            U, sig, V = ti.svd(self.F[p])
            J = 1.0

            for d in ti.static(range(self.dim)):
                J *= sig[d, d]

            #---------ANISOMPM DAMAGE---------
            if self.useAnisoMPMDamageList[p] and self.useDFG:
                #Compute Geometric Resistance
                dp = self.Dp[p]
                Dc = dp - (self.l0**2 * self.damageLaplacians[p])
                
                #Get cauchy stress and its eigenvalues and eigenvectors, then construct sigmaPlus
                kirchoff = self.kirchoff_FCR(self.F[p], U@V.transpose(), J, self.mu[p], self.la[p]) #FCR elasticity             
                e, v1, v2 = self.eigenDecomposition2D(kirchoff / J) #use my eigendecomposition, comes out as three 2D vectors
                sigmaPlus = (self.macaulay(e[0]) * v1.outer_product(v1)) + (self.macaulay(e[1]) * v2.outer_product(v2))

                #Compute Phi
                A = ti.Matrix.identity(float, self.dim) #set up structural tensor for later use
                Asig = A @ sigmaPlus
                sigA = sigmaPlus @ A
                contraction = 0.0
                for i in ti.static(range(self.dim)):
                    for j in ti.static(range(self.dim)):
                        contraction += Asig[i, j] * sigA[i, j] #TODO:3D
                phi = (1.0 / self.sigmaC[p]**2.0) * contraction

                dTilde = max(self.dTildeH[p], self.zeta * self.macaulay(phi - 1)) #make sure driving force always increasing
                self.dTildeH[p] = dTilde #update max

                diff = ((1 - dp) * dTilde) - Dc
                newD = dp + ((self.dt / self.eta) * self.macaulay(diff))
                self.Dp[p] = min(1.0, newD)

    @ti.kernel
    def momentumP2GandComputeForces(self):
        # P2G and Internal Grid Forces
        ti.block_dim(256)
        ti.no_activate(self.particle)
        particleCount = 0
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            particleCount += 1
            base = ti.floor(self.x[p] * self.invDx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            #for particle p, compute base index
            fx = self.x[p] * self.invDx - base.cast(float)
            
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]

            U, sig, V = ti.svd(self.F[p])
            J = 1.0

            for d in ti.static(range(self.dim)):
                J *= sig[d, d]
            
            mu = self.mu[p]
            la = self.la[p]
            #Compute Kirchoff Stress
            kirchoff = self.kirchoff_FCR(self.F[p], U@V.transpose(), J, mu, la)

            #NOTE Grab the sigmaMax here so we can learn how to better threshold the stress for damage
            e, v1, v2 = self.eigenDecomposition2D(kirchoff / J) #use my eigendecomposition, comes out as three 2D vectors
            self.sigmaMax[p] = e[0] if e[0] > e[1] else e[1] #TODO:3D

            #Compute E for ViYi barrier stiffness and volumetric term
            E = (mu*(3*la + 2*mu)) / (la + mu) #recompute E for this particle
            vol = self.Vp[p]

            #P2G for velocity, normals, force, and ViYi for implicit!
            for offset in ti.static(ti.grouped(self.stencil_range())): # Loop over grid node stencil
                gridIdx = base + offset
                dpos = (offset.cast(float) - fx) * self.dx
                
                weight = 1.0
                oidx = self.offset2idx(offset)
                if self.useDFG:
                    weight = self.p_cached_w[p, oidx]
                else:
                    #if we arent using DFG we havent computed weights yet!
                    for d in ti.static(range(self.dim)):
                        weight *= w[offset[d]][d]
                    self.p_cached_w[p, oidx] = weight
                
                dweight = ti.Vector.zero(float,self.dim)
                if ti.static(self.dim == 2):
                    dweight[0] = self.invDx * dw[offset[0]][0] * w[offset[1]][1]
                    dweight[1] = self.invDx * w[offset[0]][0] * dw[offset[1]][1]
                else:
                    dweight[0] = dw[offset[0]][0]*w[offset[1]][1]*w[offset[2]][2] * self.invDx
                    dweight[1] = w[offset[0]][0]*dw[offset[1]][1]*w[offset[2]][2] * self.invDx
                    dweight[2] = w[offset[0]][0]*w[offset[1]][1]*dw[offset[2]][2] * self.invDx
                
                force = -vol * kirchoff @ dweight

                #Cache wip and nabla wip for this particle and each of its 3^d grid nodes in the interpolation stencil
                self.p_cached_dw[p, oidx] = dweight #always save this because we've never computed nabla wip before!

                if self.separable[gridIdx] != 1 or self.useDFG == False: 
                    #treat node as one field
                    if(self.useAPIC):
                        self.grid_q1[gridIdx] += self.mp[p] * weight * (self.v[p] + self.C[p] @ dpos) #momentum transfer (APIC)
                    else:
                        self.grid_q1[gridIdx] += self.mp[p] * weight * self.v[p] #momentum transfer (PIC)

                    if(self.useDFG == False): #need to do this because for explicitMPM we skip the massP2G routine
                        #print("grid_m1 gridIdx:", gridIdx)
                        self.grid_m1[gridIdx] += weight * self.mp[p] #add mass to active field for this particle

                    self.grid_f1[gridIdx] += force #accumulate grid forces
                    self.grid_n1[gridIdx] += dweight * self.mp[p] #add to the normal for this field at this grid node, remember we need to normalize it later!

                elif self.separable[gridIdx] == 1 and self.useDFG == True:
                    #treat node as having two fields
                    oidx = self.offset2idx(offset)
                    fieldIdx = self.particleAF[p, oidx] #grab the field that this particle is in for this node
                    if fieldIdx == 0:
                        #field 1
                        if self.useAPIC:
                            self.grid_q1[gridIdx] += self.mp[p] * weight * (self.v[p] + self.C[p] @ dpos) #momentum transfer (APIC)
                        else:
                            self.grid_q1[gridIdx] += self.mp[p] * weight * self.v[p] #momentum transfer (PIC)
                        self.grid_f1[gridIdx] += force                    
                        self.grid_n1[gridIdx] += dweight * self.mp[p] #add to the normal for this field at this grid node, remember we need to normalize it later!
                        if not self.symplectic: self.gridViYi1[gridIdx] += vol * E * weight #Transfer ViYi to the grid for barrier function, field 1
                    elif fieldIdx == 1:
                        #field 2
                        if self.useAPIC:
                            self.grid_q2[gridIdx] += self.mp[p] * weight * (self.v[p] + self.C[p] @ dpos) #momentum transfer (APIC)
                        else:
                            self.grid_q2[gridIdx] += self.mp[p] * weight * self.v[p] #momentum transfer (PIC)
                        self.grid_f2[gridIdx] += force                    
                        self.grid_n2[gridIdx] += dweight * self.mp[p] #add to the normal for this field at this grid node, remember we need to normalize it later!
                        if not self.symplectic: self.gridViYi2[gridIdx] += vol * E * weight #Transfer ViYi to the grid for barrier function, field 2
                    else:
                        print("ERROR: why did we get here???")
                        #raise ValueError('ERROR: invalid field idx somehow')

    @ti.kernel
    def momentumToVelocity(self):
        self.activeNodes[None] = 0
        for I in ti.grouped(self.grid_m1):
            if self.grid_m1[I] > 0:
                self.grid_v1[I] = self.grid_q1[I] / self.grid_m1[I]
                self.grid_vn1[I] = self.grid_q1[I] / self.grid_m1[I]

                #Setup our grid indeces <-> DOF mapping
                idx = self.activeNodes[None].atomic_add(1)
                self.grid_idx[I] = idx
                self.dof2idx[idx] = I

            if self.separable[I] == 1:
                self.grid_v2[I] = self.grid_q2[I] / self.grid_m2[I] 
                self.grid_vn2[I] = self.grid_q2[I] / self.grid_m2[I]

    @ti.kernel
    def addGridForces(self):
        for I in ti.grouped(self.grid_m1):
            if self.grid_m1[I] > 0:
                #print("f1:", self.grid_f1[I], "m1: ", self.grid_m1[I])
                self.grid_v1[I] += (self.grid_f1[I] * self.dt) / self.grid_m1[I]
                #print("v1:", self.grid_v1[I])
            if self.separable[I] == 1:
                self.grid_v2[I] += (self.grid_f2[I] * self.dt) / self.grid_m2[I]

    @ti.kernel
    def addGravity(self):
         #Add Gravity
        for I in ti.grouped(self.grid_m1):
            if self.grid_m1[I] > 0:
                self.grid_v1[I][1] += self.dt * self.gravMag
            if self.separable[I] == 1:
                self.grid_v2[I][1] += self.dt * self.gravMag

    @ti.kernel
    def applyImpulse(self):
        for I in ti.grouped(self.grid_m1):
            dist = self.impulseCenter - (self.dx * I)
            dv = dist / (0.01 + dist.norm()) * self.impulseStrength * self.dt
            if self.symplectic:
                if self.grid_m1[I] > 0:
                    self.grid_v1[I] += dv
                if self.separable[I] == 1:
                    self.grid_v2[I] += dv
            else:
                if self.grid_m1[I] > 0:
                    self.grid_fi1[I] += dv
                if self.separable[I] == 1:
                    self.grid_fi2[I] += dv


    @ti.kernel
    def computeContactForces(self):
        #Frictional Contact Forces
        for I in ti.grouped(self.grid_m1):
            if self.separable[I] == 1: #only apply these forces to separable nodes with two fields
                #momentium
                q_1 = self.grid_v1[I] * self.grid_m1[I]
                q_2 = self.grid_v2[I] * self.grid_m2[I]
                q_cm = q_1 + q_2 

                #mass
                m_1 = self.grid_m1[I]
                m_2 = self.grid_m2[I]
                m_cm = m_1 + m_2

                #velocity
                v_1 = self.grid_v1[I]
                v_2 = self.grid_v2[I]
                v_cm = q_cm / m_cm #NOTE: we need to compute this like this to conserve mass and momentum

                #normals
                n_1 = self.grid_n1[I].normalized() #don't forget to normalize these!!
                n_2 = self.grid_n2[I].normalized()
                n_cm1 = (n_1 - n_2).normalized()
                n_cm2 = -n_cm1

                #Save these c.o.m. normals in the original storage
                self.grid_n1[I] = n_cm1
                self.grid_n2[I] = n_cm2

                if self.symplectic: #only compute these contact forces for explicit sim
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

                    if (v_cm - v_1).dot(n_cm1) + (v_cm - v_2).dot(n_cm2) < 0:
                        tanMin1 = self.fricCoeff * abs(fNormal1) if self.fricCoeff * abs(fNormal1) < abs(fTanMag1) else abs(fTanMag1)
                        f_c1 += (fNormal1 * n_cm1) + (tanMin1 * fTanSign1 * tanDirection1)
                        tanMin1 = self.fricCoeff * abs(fNormal2) if self.fricCoeff * abs(fNormal2) < abs(fTanMag2) else abs(fTanMag2)
                        f_c2 += (fNormal2 * n_cm2) + (tanMin1 * fTanSign2 * tanDirection2)

                    #Now save these forces for later
                    self.grid_fct1[I] = f_c1
                    self.grid_fct2[I] = f_c2

    @ti.kernel
    def addContactForces(self):        
        #Add Friction Forces
        for I in ti.grouped(self.grid_m1):    
            if self.separable[I] == 1:
                if(self.useFrictionalContact and self.symplectic):
                    self.grid_v1[I] += (self.grid_fct1[I] / self.grid_m1[I]) * self.dt # use field 1 force to update field 1 particles (for green nodes, ie separable contact)
                    self.grid_v2[I] += (self.grid_fct2[I] / self.grid_m2[I]) * self.dt # use field 2 force to update field 2 particles
    
    #----IMPLICIT-METHODS-BEGIN-----------------------------------------------------
    @ti.kernel
    def BackupStrain(self):
        for i in range(self.numParticles):
            self.F_backup[i] = self.F[i]

    @ti.kernel
    def RestoreStrain(self):
        for i in range(self.numParticles):
            self.F[i] = self.F_backup[i]
    
    #Set initial guess for Newton iteration
    @ti.kernel
    def BuildInitialBoundary(self):
        ndof = self.activeNodes[None]
        sdof = self.separableNodes[None]
        for i in range(ndof):
            if self.boundary[i] == 1: 
                #if node is in a boundary, guess 0
                for d in ti.static(range(self.dim)): #TODO:Slip and Moving Boundary
                    self.dv[i*self.dim+d] = 0
            else:                      
                #if not in boundary, guess change in velocity from gravity
                for d in ti.static(range(self.dim)):
                    self.dv[i*self.dim+d] = self.gravity[None][d]*self.dt
        if self.useDFG:
            for i in range(sdof):
                if self.boundary[ndof + i] == 1:
                    #if node is in a boundary, guess 0
                    for d in ti.static(range(self.dim)):
                        self.dv[ndof*self.dim + i*self.dim + d] = 0
                else:
                    #if not in boundary, guess change in velocity from gravity
                    for d in ti.static(range(self.dim)):
                        self.dv[ndof*self.dim + i*self.dim + d] = self.gravity[None][d] * self.dt

    @ti.kernel
    def UpdateDV(self, alpha:float):
        ndof = self.activeNodes[None]
        sdof = self.separableNodes[None] if self.useDFG else 0
        for i in range((ndof + sdof) * self.dim):
            self.DV[i] = self.dv[i] + self.data_x[i] * alpha

    @ti.kernel
    def UpdateState(self):
        # Update deformation gradient F = (I + deltat * d(v'))F
        # where v' = v + DV
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.x[p] * self.invDx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            new_F = ti.Matrix.zero(float, self.dim, self.dim)
            # loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                g_v1 = self.grid_v1[base + offset]
                g_v2 = self.grid_v2[base + offset]
                g_dof1 = self.grid_idx[base + offset]
                g_dof2 = self.grid_sep_idx[base + offset]
                
                #grab cached dweight for this p -> i pairing
                oidx = self.offset2idx(offset)
                dweight = self.p_cached_dw[p, oidx]

                DV = ti.Vector.zero(float, self.dim)
                if self.separable[base + offset] != 1:
                    #single field node
                    for d in ti.static(range(self.dim)):
                        DV[d] = self.DV[g_dof1 * self.dim + d]
                    g_v1 += DV
                    new_F += g_v1.outer_product(dweight)
                else:
                    #2-field node
                    fieldIdx = self.particleAF[p, oidx]
                    if fieldIdx == 0:
                        for d in ti.static(range(self.dim)):
                            DV[d] = self.DV[g_dof1 * self.dim + d]
                        g_v1 += DV
                        new_F += g_v1.outer_product(dweight)
                    else:
                        ndof = self.activeNodes[None]
                        for d in ti.static(range(self.dim)):
                            DV[d] = self.DV[ndof*self.dim + g_dof2*self.dim + d]
                        g_v2 += DV
                        new_F += g_v2.outer_product(dweight)
            self.F[p] = (ti.Matrix.identity(float, self.dim) + self.dt * new_F) @ self.F[p]

    #Compute Total Energy
    @ti.kernel
    def TotalEnergy(self) -> float:
        #elastic energy
        ee = 0.0
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            F = self.F[p]
            la, mu = self.la[p], self.mu[p]
            U, sig, V = svd(F)
            psi = elasticity_energy(sig, la, mu)
            ee += self.Vp[p] * psi
        
        #kinetic energy
        ke = 0.0
        for I in ti.grouped(self.grid_m1):
            if self.grid_m1[I] > 0:
                i = self.grid_idx[I]
                dv = ti.Vector.zero(float, self.dim)
                for d in ti.static(range(self.dim)):
                    dv[d] = self.DV[i*self.dim+d]
                ke += dv.dot(dv) * self.grid_m1[I]
            if self.separable[I] == 1:
                #also compute KE from second field if separable
                i = self.grid_sep_idx[I]
                ndof = self.activeNodes[None]
                dv = ti.Vector.zero(float, self.dim)
                for d in ti.static(range(self.dim)):
                    dv[d] = self.DV[ndof*self.dim + i*self.dim + d]
                ke += dv.dot(dv) * self.grid_m2[I] 

        #gravitational energy
        ge = 0.0
        for I in ti.grouped(self.grid_m1):
            if self.grid_m1[I] > 0:
                i = self.grid_idx[I]
                dv = ti.Vector.zero(float, self.dim)
                for d in ti.static(range(self.dim)):
                    dv[d] = self.DV[i*self.dim+d]
                ge -= self.dt * dv.dot(self.gravity[None]) * self.grid_m1[I]
            if self.separable[I] == 1:
                #also compute GE from second field if separable
                i = self.grid_sep_idx[I]
                ndof = self.activeNodes[None]
                dv = ti.Vector.zero(float, self.dim)
                for d in ti.static(range(self.dim)):
                    dv[d] = self.DV[ndof*self.dim + i*self.dim + d] #grabbing from second field portion of DV
                ge -= self.dt * dv.dot(self.gravity[None]) * self.grid_m2[I]

        #impulse energy
        ie = 0.0
        if self.useImpulse:
            for I in ti.grouped(self.grid_m1):
                if self.grid_m1[I] > 0:
                    i = self.grid_idx[I]
                    dv = ti.Vector.zero(float, self.dim)
                    for d in ti.static(range(self.dim)):
                        dv[d] = self.DV[i*self.dim+d]
                    ie -= dv.dot(self.grid_fi1[I]) * self.grid_m1[I] #dt is included in grid_fi1
                if self.separable[I] == 1:
                    #also compute IE from second field if separable
                    i = self.grid_sep_idx[I]
                    ndof = self.activeNodes[None]
                    dv = ti.Vector.zero(float, self.dim)
                    for d in ti.static(range(self.dim)):
                        dv[d] = self.DV[ndof*self.dim + i*self.dim + d] #grabbing from second field portion of DV
                    ie -= dv.dot(self.grid_fi2[I]) * self.grid_m2[I] #dt is included in grid_fi2

        #Barrier Energy (Frictional Contact) -- NOTE: only for Separable Nodes, TAG=Barrier
        be = 0.0
        for I in ti.grouped(self.grid_m1):
            if self.separable[I] == 1:
                ci = self.gridCi[I]         #NOTE we should have computed ci by now since we do it after each UpdateDV
                chat = self.chat
                B = self.computeB(ci, chat) #takes care of the zero case too (ci >= chat)
                ViYi1 = self.gridViYi1[I]
                ViYi2 = self.gridViYi2[I]

                be += (ViYi1 + ViYi2) * B
   
        return ee + ke / 2 + ge + ie + be

    #Compute Force based on candidate F; NOTE: this is only used for implicit
    @ti.kernel
    def computeForces(self):
        # force is computed more than once in Newton iteration 
        # temporarily set all grid force to zero
        for I in ti.grouped(self.grid_m1):
            if self.grid_m1[I] > 0:  # No need for epsilon here
                self.grid_f1[I] = ti.Vector.zero(float, self.dim)
            if self.separable[I] == 1:
                self.grid_f2[I] = ti.Vector.zero(float, self.dim)

        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.x[p] * self.invDx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            mu, la = self.mu[p], self.la[p]
            P = elasticity_first_piola_kirchoff_stress(self.F[p], la, mu)
            kirchoff = P @ self.F_backup[p].transpose()

            for offset in ti.static(ti.grouped(self.stencil_range())):
                oidx = self.offset2idx(offset)
                dweight = self.p_cached_dw[p, oidx]
                force = -self.Vp[p] * kirchoff @ dweight

                if self.separable[base + offset] != 1:
                    self.grid_f1[base + offset] += force
                else:
                    fieldIdx = self.particleAF[p, oidx] #grab the field that this particle is in for this node
                    if fieldIdx == 0:
                        self.grid_f1[base + offset] += force
                    elif fieldIdx == 1:
                        self.grid_f2[base + offset] += force
                    else:
                        print("[Compute Implicit Force] ERROR: why did we get here???")

    @ti.kernel
    def ComputeResidual(self, project:ti.template()):
        ndof = self.activeNodes[None]
        for i in ti.ndrange(ndof):
            gid = self.dof2idx[i]
            m = self.grid_m1[gid]
            f = self.grid_f1[gid]
            impulse = self.grid_fi1[gid] #this is a * dt = velocity
            for d in ti.static(range(self.dim)):
                self.rhs[i*self.dim+d] = (self.DV[i*self.dim+d] * m) - (self.dt * f[d]) - (self.dt * m * self.gravity[None][d]) - (impulse[d] * m) #NOTE: here we incorporate all external forces (gravity and impulses)
        
        #if using DFG, iterate the separable nodes to set rhs for field 2
        if self.useDFG:
            sdof = self.separableNodes[None]
            for i in range(sdof):
                gid2 = self.sepDof2idx[i]
                m2 = self.grid_m2[gid2]
                f2 = self.grid_f2[gid2]
                impulse2 = self.grid_fi2[gid2] #this is a * dt = velocity
                for d in ti.static(range(self.dim)):
                    self.rhs[ndof*self.dim + i*self.dim + d] = (self.DV[ndof*self.dim + i*self.dim + d] * m2) - (self.dt * f2[d]) - (self.dt * m2 * self.gravity[None][d]) - (impulse2[d] * m2)

        #Add Barrier Energy Gradient to RHS for separable nodes, TAG=Barrier
        if self.useDFG:
            for I in ti.grouped(self.grid_m1):
                if self.separable[I] == 1:
                    i = self.grid_idx[I]
                    i2 = self.grid_sep_idx[I]

                    #barrier function derivative
                    bPrime = self.computeBPrime(self.gridCi[I], self.chat)
                    bPrime *= (self.gridViYi1[I] + self.gridViYi2[I]) #multiply by the total ViYi contributions to this node
                    
                    n_cm1 = self.grid_n1[I]
                    nablaB1 = bPrime * -n_cm1 #times nabla ci
                    nablaB2 = bPrime * n_cm1  #times nabla ci

                    for d in ti.static(range(self.dim)):
                        self.rhs[i*self.dim + d] += nablaB1[d]                  #add field 1 to first part of rhs
                        self.rhs[ndof*self.dim + i2*self.dim + d] += nablaB2[d] #add field 2 to second part of rhs

        if project == True:
            # Boundary projection
            ndof = self.activeNodes[None]
            for i in range(ndof):
                if self.boundary[i] == 1:
                    for d in ti.static(range(self.dim)):
                        self.rhs[i * self.dim + d] = 0 #TODO:Slip and Moving Boundaries
            #if using DFG we also project field 2 entries
            if self.useDFG:
                sdof = self.separableNodes[None]
                for i in range(sdof):
                    if self.boundary[ndof + i] == 1:
                        for d in ti.static(range(self.dim)):
                            self.rhs[ndof*self.dim + i*self.dim + d] = 0 #TODO:Slip and Moving Boundaries

    @ti.func
    def computedFdX(self, dPdF, wi, wj):
        dFdX = ti.Matrix.zero(float, self.dim, self.dim)
        dFdX[0,0] = dPdF[0+0,0+0]*wi[0]*wj[0] + dPdF[2+0,0+0]*wi[1]*wj[0] + dPdF[0+0,2+0]*wi[0]*wj[1] + dPdF[2+0,2+0]*wi[1]*wj[1]
        dFdX[0,1] = dPdF[0+0,0+1]*wi[0]*wj[0] + dPdF[2+0,0+1]*wi[1]*wj[0] + dPdF[0+0,2+1]*wi[0]*wj[1] + dPdF[2+0,2+1]*wi[1]*wj[1]
        dFdX[1,0] = dPdF[0+1,0+0]*wi[0]*wj[0] + dPdF[2+1,0+0]*wi[1]*wj[0] + dPdF[0+1,2+0]*wi[0]*wj[1] + dPdF[2+1,2+0]*wi[1]*wj[1]
        dFdX[1,1] = dPdF[0+1,0+1]*wi[0]*wj[0] + dPdF[2+1,0+1]*wi[1]*wj[0] + dPdF[0+1,2+1]*wi[0]*wj[1] + dPdF[2+1,2+1]*wi[1]*wj[1]

        return dFdX

    @ti.func
    def computedFdX3D(self, dPdF, wi, wj):
        dFdX = ti.Matrix.zero(float, self.dim, self.dim)

        dFdX[0,0] = dPdF[0+0,0+0]*wi[0]*wj[0]+dPdF[3+0,0+0]*wi[1]*wj[0]+dPdF[6+0,0+0]*wi[2]*wj[0]+dPdF[0+0,3+0]*wi[0]*wj[1]+dPdF[3+0,3+0]*wi[1]*wj[1]+dPdF[6+0,3+0]*wi[2]*wj[1]+dPdF[0+0,6+0]*wi[0]*wj[2]+dPdF[3+0,6+0]*wi[1]*wj[2]+dPdF[6+0,6+0]*wi[2]*wj[2]
        dFdX[1,0] = dPdF[0+1,0+0]*wi[0]*wj[0]+dPdF[3+1,0+0]*wi[1]*wj[0]+dPdF[6+1,0+0]*wi[2]*wj[0]+dPdF[0+1,3+0]*wi[0]*wj[1]+dPdF[3+1,3+0]*wi[1]*wj[1]+dPdF[6+1,3+0]*wi[2]*wj[1]+dPdF[0+1,6+0]*wi[0]*wj[2]+dPdF[3+1,6+0]*wi[1]*wj[2]+dPdF[6+1,6+0]*wi[2]*wj[2]
        dFdX[2,0] = dPdF[0+2,0+0]*wi[0]*wj[0]+dPdF[3+2,0+0]*wi[1]*wj[0]+dPdF[6+2,0+0]*wi[2]*wj[0]+dPdF[0+2,3+0]*wi[0]*wj[1]+dPdF[3+2,3+0]*wi[1]*wj[1]+dPdF[6+2,3+0]*wi[2]*wj[1]+dPdF[0+2,6+0]*wi[0]*wj[2]+dPdF[3+2,6+0]*wi[1]*wj[2]+dPdF[6+2,6+0]*wi[2]*wj[2]
        dFdX[0,1] = dPdF[0+0,0+1]*wi[0]*wj[0]+dPdF[3+0,0+1]*wi[1]*wj[0]+dPdF[6+0,0+1]*wi[2]*wj[0]+dPdF[0+0,3+1]*wi[0]*wj[1]+dPdF[3+0,3+1]*wi[1]*wj[1]+dPdF[6+0,3+1]*wi[2]*wj[1]+dPdF[0+0,6+1]*wi[0]*wj[2]+dPdF[3+0,6+1]*wi[1]*wj[2]+dPdF[6+0,6+1]*wi[2]*wj[2]
        dFdX[1,1] = dPdF[0+1,0+1]*wi[0]*wj[0]+dPdF[3+1,0+1]*wi[1]*wj[0]+dPdF[6+1,0+1]*wi[2]*wj[0]+dPdF[0+1,3+1]*wi[0]*wj[1]+dPdF[3+1,3+1]*wi[1]*wj[1]+dPdF[6+1,3+1]*wi[2]*wj[1]+dPdF[0+1,6+1]*wi[0]*wj[2]+dPdF[3+1,6+1]*wi[1]*wj[2]+dPdF[6+1,6+1]*wi[2]*wj[2]
        dFdX[2,1] = dPdF[0+2,0+1]*wi[0]*wj[0]+dPdF[3+2,0+1]*wi[1]*wj[0]+dPdF[6+2,0+1]*wi[2]*wj[0]+dPdF[0+2,3+1]*wi[0]*wj[1]+dPdF[3+2,3+1]*wi[1]*wj[1]+dPdF[6+2,3+1]*wi[2]*wj[1]+dPdF[0+2,6+1]*wi[0]*wj[2]+dPdF[3+2,6+1]*wi[1]*wj[2]+dPdF[6+2,6+1]*wi[2]*wj[2]         
        dFdX[0,2] = dPdF[0+0,0+2]*wi[0]*wj[0]+dPdF[3+0,0+2]*wi[1]*wj[0]+dPdF[6+0,0+2]*wi[2]*wj[0]+dPdF[0+0,3+2]*wi[0]*wj[1]+dPdF[3+0,3+2]*wi[1]*wj[1]+dPdF[6+0,3+2]*wi[2]*wj[1]+dPdF[0+0,6+2]*wi[0]*wj[2]+dPdF[3+0,6+2]*wi[1]*wj[2]+dPdF[6+0,6+2]*wi[2]*wj[2]              
        dFdX[1,2] = dPdF[0+1,0+2]*wi[0]*wj[0]+dPdF[3+1,0+2]*wi[1]*wj[0]+dPdF[6+1,0+2]*wi[2]*wj[0]+dPdF[0+1,3+2]*wi[0]*wj[1]+dPdF[3+1,3+2]*wi[1]*wj[1]+dPdF[6+1,3+2]*wi[2]*wj[1]+dPdF[0+1,6+2]*wi[0]*wj[2]+dPdF[3+1,6+2]*wi[1]*wj[2]+dPdF[6+1,6+2]*wi[2]*wj[2]
        dFdX[2,2] = dPdF[0+2,0+2]*wi[0]*wj[0]+dPdF[3+2,0+2]*wi[1]*wj[0]+dPdF[6+2,0+2]*wi[2]*wj[0]+dPdF[0+2,3+2]*wi[0]*wj[1]+dPdF[3+2,3+2]*wi[1]*wj[1]+dPdF[6+2,3+2]*wi[2]*wj[1]+dPdF[0+2,6+2]*wi[0]*wj[2]+dPdF[3+2,6+2]*wi[1]*wj[2]+dPdF[6+2,6+2]*wi[2]*wj[2]

        return dFdX

    @ti.func
    def constructBarrierHessianBlocks(self, hessianB):
        m00 = ti.Matrix.zero(float, self.dim, self.dim)
        m01 = ti.Matrix.zero(float, self.dim, self.dim)
        m10 = ti.Matrix.zero(float, self.dim, self.dim)
        m11 = ti.Matrix.zero(float, self.dim, self.dim)

        for i in ti.static(range(self.dim)):
            for j in ti.static(range(self.dim)):
                m00[i,j] = hessianB[i,j]
                m01[i,j] = hessianB[i, j + self.dim]
                m10[i,j] = hessianB[i + self.dim, j]
                m11[i,j] = hessianB[i + self.dim, j + self.dim]

        return m00, m01, m10, m11

    # Build Matrix: construct the Hessian of Energy (w.r.t. dv)
    @ti.kernel
    def BuildMatrix(self, project:ti.template(), project_pd:ti.template()):
        nNbr = 25
        midNbr = 12
        if self.dim == 3:
            nNbr = 125
            midNbr = 62
        
        ndof = self.activeNodes[None]
        sdof = self.separableNodes[None] if self.useDFG else 0

        #Need more space if using DFG
        if self.useDFG:
            nNbr *= 2 #NOTE: we'll use this extra space as a scratch space for second field contributions
            #NOTE: leave mid number the same because we want the diagonal to be there in the first field still
            nNbr += sdof #add space for our barrier contributions to separable nodes

        #Set inertial term of hessian (diagonal of masses)
        for i in ti.ndrange(ndof):
            #initialize entries
            for j in range(nNbr):
                self.entryCol[i*nNbr + j] = -1
                self.entryVal[i*nNbr + j] = ti.Matrix.zero(float, self.dim, self.dim)
            #set diagonal
            gid = self.dof2idx[i]
            m = self.grid_m1[gid]
            self.entryCol[i*nNbr + midNbr] = i
            self.entryVal[i*nNbr + midNbr] = ti.Matrix.identity(float, self.dim) * m
        for i in ti.ndrange(sdof):
            #initialize entries
            for j in range(nNbr):
                self.entryCol[(ndof + i)*nNbr + j] = -1
                self.entryVal[(ndof + i)*nNbr + j] = ti.Matrix.zero(float, self.dim, self.dim)
            #set diagonal
            gid2 = self.sepDof2idx[i]
            m2 = self.grid_m2[gid2]
            self.entryCol[(ndof + i)*nNbr + midNbr] = ndof + i #should be total dof index here!
            self.entryVal[(ndof + i)*nNbr + midNbr] = ti.Matrix.identity(float, self.dim) * m2

        # Now add the second term of Hessian
        for I in ti.grouped(self.pid):
            p = self.pid[I]

            vol = self.Vp[p]
            F = self.F[p]
            mu, la = self.mu[p], self.la[p]
            dPdF = elasticity_first_piola_kirchoff_stress_derivative(F, la, mu, project_pd)
            
            base = ti.floor(self.x[p] * self.invDx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            if ti.static(self.dim == 2):
                #Cache the proper indeces to map this particle to each of its 3^dim stencil nodes
                for offset in ti.static(ti.grouped(self.stencil_range())):
                    oidx = self.offset2idx(offset)
                    if not self.useDFG or self.separable[base + offset] != 1:
                        self.p_cached_idx[p,oidx] = self.grid_idx[base + offset]
                    else:
                        fieldIdx = self.particleAF[p, oidx]
                        if fieldIdx == 0:
                            self.p_cached_idx[p, oidx] = self.grid_idx[base + offset]
                        elif fieldIdx == 1:
                            self.p_cached_idx[p, oidx] = ndof + self.grid_sep_idx[base + offset] #ndof + the 0-indexed seperable node idx
                        else:
                            ValueError("ERROR: fieldIdx not set yet :(")
                
                #Compute and store col and val based on these indeces for p
                for i in range(3**self.dim):
                    wi = self.F_backup[p].transpose() @ self.p_cached_dw[p,i]
                    dofi = self.p_cached_idx[p,i]
                    nodei = self.idx2offset(i)
                    for j in range(3**self.dim):
                        wj = self.F_backup[p].transpose() @ self.p_cached_dw[p,j]
                        dofj = self.p_cached_idx[p,j]
                        nodej = self.idx2offset(j)
                    
                        dFdX = self.computedFdX(dPdF, wi, wj)
                        dFdX = dFdX * vol * self.dt * self.dt

                        doffs = self.linear_offset(nodei - nodej)
                        if dofj >= ndof: doffs += 25 #if column index is an sdof, give the offset + 25 to be in second half of the row (our scratch space we set up)
                        ioffset = dofi*nNbr + doffs
                        
                        self.entryCol[ioffset] = dofj
                        self.entryVal[ioffset] += dFdX
            
            if ti.static(self.dim == 3):
                #Cache the proper indeces to map this particle to each of its 3^dim stencil nodes
                for offset in ti.static(ti.grouped(self.stencil_range())):
                    oidx = self.offset2idx(offset)
                    if not self.useDFG or self.separable[base + offset] != 1:
                        self.p_cached_idx[p,oidx] = self.grid_idx[base + offset]
                    else:
                        fieldIdx = self.particleAF[p, oidx]
                        if fieldIdx == 0:
                            self.p_cached_idx[p, oidx] = self.grid_idx[base + offset]
                        elif fieldIdx == 1:
                            self.p_cached_idx[p, oidx] = ndof + self.grid_sep_idx[base + offset] #ndof + the 0-indexed seperable node idx
                        else:
                            ValueError("ERROR: fieldIdx not set yet :(")
                
                #Compute and store col and val based on these indeces for p
                for i in range(3**self.dim):
                    wi = self.F_backup[p].transpose() @ self.p_cached_dw[p,i]
                    dofi = self.p_cached_idx[p,i]
                    nodei = self.idx2offset(i)
                    for j in range(3**self.dim):
                        wj = self.F_backup[p].transpose() @ self.p_cached_dw[p,j]
                        dofj = self.p_cached_idx[p,j]
                        nodej = self.idx2offset(j)
                    
                        dFdX = self.computedFdX3D(dPdF, wi, wj)
                        dFdX = dFdX * vol * self.dt * self.dt

                        doffs = self.linear_offset3D(nodei - nodej)
                        if dofj >= ndof: doffs += 125 #if column index is an sdof, give the offset + 25 to be in second half of the row (our scratch space we set up)
                        ioffset = dofi*nNbr + doffs

                        self.entryCol[ioffset] = dofj
                        self.entryVal[ioffset] += dFdX  

        #Add Barrier Energy Hessian Terms, TAG=Barrier
        if self.useDFG:
            for I in ti.grouped(self.grid_m1):
                if self.separable[I] == 1:
                    dof1 = self.grid_idx[I]
                    dof2 = ndof + self.grid_sep_idx[I]
                    i2 = self.grid_sep_idx[I]

                    #barrier function second derivative
                    bDoublePrime = self.computeBDoublePrime(self.gridCi[I], self.chat)
                    bDoublePrime *= (self.gridViYi1[I] + self.gridViYi2[I]) #multiply this contribution

                    #stack nablaC in a (dim*2) x 1 vector
                    n_cm1 = self.grid_n1[I]
                    nablaC = ti.Vector.zero(float, self.dim * 2)
                    for d in ti.static(range(self.dim)):
                        nablaC[d] = -n_cm1[d]
                        nablaC[d + self.dim] = n_cm1[d]

                    hessianB =  bDoublePrime * nablaC.outer_product(nablaC) #this hessian is (dim*2) by (dim*2)

                    #grab the four block wise pieces, each being dim by dim
                    m00, m01, m10, m11 = self.constructBarrierHessianBlocks(hessianB)

                    #top left block (on diagonal)
                    self.entryCol[dof1*nNbr + midNbr] = dof1
                    self.entryVal[dof1*nNbr + midNbr] += m00

                    #top right block (off diagonal)
                    self.entryCol[dof1*nNbr + (nNbr - sdof) + i2] = dof2 #TODO:Barrier these might be switched ??
                    self.entryVal[dof1*nNbr + (nNbr - sdof) + i2] += m01

                    #bottom left block (off diagonal)
                    self.entryCol[dof2*nNbr + (nNbr - sdof) + i2] = dof1 #TODO:Barrier these might be switched ??
                    self.entryVal[dof2*nNbr + (nNbr - sdof) + i2] += m10

                    #bottom right block (also on diagonal)
                    self.entryCol[dof2*nNbr + midNbr] = dof2
                    self.entryVal[dof2*nNbr + midNbr] += m11

    #Set up and solve the linear system using PCG and Sparse Matrix
    def SolveLinearSystem(self):
        ndof = self.activeNodes[None]
        sdof = self.separableNodes[None] if self.useDFG else 0
        nNbr = int(5**self.dim)
        if self.useDFG:
            nNbr *= 2
            nNbr += sdof
        if self.dim == 2:
            self.matrix.prepareColandVal(ndof + sdof)
            self.matrix.setFromColandValDFG(self.entryCol, self.entryVal, ndof + sdof, nNbr)
        else:
            self.matrix.prepareColandVal(ndof + sdof, d = 3)
            self.matrix.setFromColandVal3DFG(self.entryCol, self.entryVal, ndof + sdof, nNbr)

        #print(self.matrix.toFullMatrix())

        self.cgsolver.compute(self.matrix, stride = self.dim)
        self.cgsolver.setBoundary(self.boundary) #TODO:Slip and Moving Boundaries
        self.cgsolver.solve(self.rhs, True) #second param is whether to print CG convergence updates

        for i in range((ndof + sdof) * self.dim):
            self.data_x[i] = -self.cgsolver.x[i]

    #Fill dv with the portion of data_x that we determined reduced the energy
    @ti.kernel
    def LineSearch(self, alpha:float):
        ndof = self.activeNodes[None]
        sdof = self.separableNodes[None] if self.useDFG else 0
        for i in range((ndof + sdof) * self.dim):
            self.dv[i] += self.data_x[i] * alpha

    @ti.kernel
    def implicitUpdate(self):
        ndof = self.activeNodes[None]
        for i in range(ndof):
            dv = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                dv[d] = self.dv[i*self.dim+d]
            gid = self.dof2idx[i]
            self.grid_v1[gid] += dv
        if self.useDFG:
            for i in range(self.separableNodes[None]):
                dv2 = ti.Vector.zero(float, self.dim)
                for d in ti.static(range(self.dim)):
                    dv2[d] = self.dv[ndof*self.dim + i*self.dim + d]
                gid2 = self.sepDof2idx[i]
                self.grid_v2[gid2] += dv2

    #Compute our constraint for each separable grid node #NOTE: remember to do this every time we update DV!!!
    @ti.kernel
    def computeCi(self):
        for I in ti.grouped(self.grid_m1):
            if self.separable[I] == 1:
                #Grab dof indeces
                i = self.grid_idx[I]
                i2 = self.grid_sep_idx[I]

                #Grab vn for each field
                v1n = self.grid_v1[I]
                v2n = self.grid_v2[I]

                #Now construct dv1 and dv2
                ndof = self.activeNodes[None]
                dv1 = ti.Vector.zero(float, self.dim)
                dv2 = ti.Vector.zero(float, self.dim)
                for d in ti.static(range(self.dim)):
                    dv1[d] = self.DV[i*self.dim + d] #grab from first portion of DV
                    dv2[d] = self.DV[ndof*self.dim + i2*self.dim + d] #grabbing from second field portion of DV
                
                #Grab n1cm
                n_cm1 = self.grid_n1[I] #NOTE: we stored n_cm1 in here during computeContactForces
                
                #Set ci based on constraint
                self.gridCi[I] = (v2n + dv2 - v1n - dv1).dot(n_cm1)

    @ti.kernel
    def projectCi(self):
        #Now we must ensure that there are no Ci <= 0, TAG=Barrier
        for I in ti.grouped(self.grid_m1):
            if self.separable[I] == 1:
                ci = self.gridCi[I]
                if ci > 0: #constraint already satisfied
                    print("constraints violated at beginning")
                    continue
                i = self.grid_idx[I]
                i2 = self.grid_sep_idx[I]

                #TODO:Barrier

    @ti.kernel
    def checkConstraints(self):
        self.constraintsViolated[None] = 0
        for i in range(self.separableNodes[None]):
            I = self.sepDof2idx[i]
            ci = self.gridCi[I]
            if ci <= 0:
                self.constraintsViolated[None] = 1

    def implicitNewton(self):
        printProgress = True
                
        self.BackupStrain() #Backup F because we need to iteratively check how the new candidate velocities are affecting F (to compute energy)
        
        self.data_x.fill(0)
        self.rhs.fill(0)
        self.dv.fill(0)
        self.DV.fill(0)
        self.BuildInitialBoundary() #initial guess based on boundaries

        #Make sure no constraints are broken (ensure our guess makes all c_i > 0), TAG=Barrier
        self.UpdateDV(0.0) #update DV to hold what we just stored in dv as init guess (need this for computeCi)
        self.computeCi() #compute all constraints and save them
        self.projectCi() #fix any broken constraints in our initial guess

        #Newton Iteration
        max_iter_newton = 150 #NOTE: should be inf, 10k enough
        for iter in range(max_iter_newton):
            #print("[Newton] Iteration", iter)
            self.RestoreStrain() #Reload original F
            self.rhs.fill(0)
            self.UpdateDV(0.0) #set DV_i = dv_i
            self.computeCi() #do this every time we update DV
            self.UpdateState() #compute updated F based on DV
            E0 = self.TotalEnergy() #compute total energy based on candidate DV
            #print("[Newton] E0 = ", E0) #NOTE: got same E0 using useDFG = True and False!
            self.computeForces() #Compute grid forces based on the candidate Fs
            self.ComputeResidual(True) #Compute g, True -> projectResidual

            #Check if our guess was enough, NOTE: good for freefall
            if iter == 0:
                ndof = self.activeNodes[None]
                sdof = self.separableNodes[None] if self.useDFG else 0
                rnorm = np.linalg.norm(self.rhs.to_numpy()[0:(ndof + sdof)*self.dim])
                if rnorm < 1e-8:
                    if printProgress: print("[Newton] Newton finished in", iter, "iteration(s) with residual", rnorm)
                    break

            self.BuildMatrix(True, False) # Compute H
            self.SolveLinearSystem()  # solve dx = H^(-1) g

            # Exiting Newton
            ndof = self.activeNodes[None]
            sdof = self.separableNodes[None] if self.useDFG else 0
            rnorm = np.linalg.norm(self.rhs.to_numpy()[0:(ndof + sdof)*self.dim])
            #print("norm", rnorm)
            ddvnorm = np.linalg.norm(self.data_x.to_numpy()[0:(ndof + sdof)*self.dim], np.inf) #NOTE: IPC convergence criterion
            #print("ddvnorm", ddvnorm)
            if ddvnorm < 1e-3:
                if printProgress: print("[Newton] Newton finished in", iter, "iteration(s) with residual", ddvnorm)
                break

            #Line Search: what alpha gives the best reduction in energy?
            alpha = 1.0
            #TODO:Barrier - compute alpha_CCD that ensures all c_i > 0
            for _ in range(15): #NOTE: 
                self.RestoreStrain()    #reload original F again
                self.UpdateDV(alpha)    #DV_i = dv_i + alpha * data_x_i
                self.computeCi()        #recompute every time we update DV
                self.UpdateState()      #Compute updated F based on DV
                
                #TODO:Barrier - may result in tunneling, TAG=Barrier
                self.checkConstraints()
                if self.constraintsViolated[None] == 1:
                    alpha /= 2.0
                    continue
                
                E = self.TotalEnergy()  #compute energy based on F
                print("alpha=", alpha, "E=", E)
                if E <= E0:             #If the energy we found is less than original, we're good!
                    break
                alpha /= 2              #split alpha in half each time
            if alpha == 1/2**15:
                print("[Line Search] ERROR: Check the direction!")
                alpha = 0.0
                break
            #print("[Line Search] Finished with Alpha:", alpha, ", E:", E)
            self.LineSearch(alpha)

            self.RestoreStrain()

        if iter == max_iter_newton - 1:
            if printProgress: print("[Newton] Max iteration reached! Current iter:", max_iter_newton-1, "with residual:", rnorm)
        
        self.implicitUpdate()
        self.RestoreStrain()

    #Iterate particles to compute maximum velocity
    @ti.kernel
    def getMaxVelocity(self)-> float:
        maxV = self.v[0]
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            #Compute max v in each dimension
            for d in ti.static(range(self.dim)):
                ti.atomic_max(maxV[d], self.v[p][d])
        return ti.sqrt(maxV.norm_sqr())

    #-----IMPLICIT-METHODS-END-----------------------------------------------------

    @ti.kernel
    def G2P(self):
        # grid to particle (G2P)
        ti.block_dim(256)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.x[p] * self.invDx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)
            fx = self.x[p] * self.invDx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]
            new_v_PIC = ti.Vector.zero(float, self.dim) #contain PIC velocity
            new_v_FLIP = ti.Vector.zero(float, self.dim) #contain FLIP velocity
            new_v = ti.Vector.zero(float, self.dim) #contain the blend velocity
            new_C = ti.Matrix.zero(float, self.dim, self.dim)
            new_F = ti.Matrix.zero(float, self.dim, self.dim)
            for offset in ti.static(ti.grouped(self.stencil_range())): # Loop over grid node stencil
                dpos = offset.cast(float) - fx
                gridIdx = base + offset
                g_v_n = self.grid_vn1[gridIdx] #v_i^n field 1
                g_v2_n = self.grid_vn2[gridIdx] #v_i^n field 2
                g_v_np1 = self.grid_v1[gridIdx] #v_i^n+1 field 1
                g_v2_np1 = self.grid_v2[gridIdx] #v_i^n+1 field 2
                
                #Grab cached weight and dweight for this p -> i pairing
                oidx = self.offset2idx(offset)
                weight = self.p_cached_w[p, oidx]
                dweight = self.p_cached_dw[p, oidx]

                if self.separable[gridIdx] != 1:
                    #treat as one field
                    new_v_PIC += weight * g_v_np1
                    new_v_FLIP += weight * (g_v_np1 - g_v_n)
                    new_C += 4 * self.invDx * weight * g_v_np1.outer_product(dpos)
                    new_F += g_v_np1.outer_product(dweight)
                else:
                    #node has two fields so choose the correct velocity contribution from the node
                    fieldIdx = self.particleAF[p, oidx] #grab the field that this particle is in for this node
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
            # new_v_FLIP *= self.dt
            # new_v_FLIP += self.v[p]

            #Compute the blend
            new_v = (self.flipPicRatio * new_v_FLIP) + ((1.0 - self.flipPicRatio) * new_v_PIC)

            self.v[p], self.C[p] = new_v, new_C #set v_p n+1 to be the blended velocity

            #print("x[p]: ", self.x[p], "newvPIC:", new_v_PIC)
            new_v_PIC *= self.dt
            self.x[p] += new_v_PIC # advection, use PIC velocity for advection regardless of PIC, FLIP, or APIC
            #self.x[p] += self.dt * new_v_PIC # advection, use PIC velocity for advection regardless of PIC, FLIP, or APIC
            
            new_F *= self.dt
            self.F[p] = (ti.Matrix.identity(float, self.dim) + new_F) @ self.F[p] #updateF (explicitMPM way)
            #self.F[p] = (ti.Matrix.identity(float, self.dim) + (self.dt * new_F)) @ self.F[p] #updateF (explicitMPM way)

    #------------Collision Objects---------------

    #update collision object centers based on the translation and velocity
    @ti.kernel
    def updateCollisionObjects(self, id: ti.i32):
        self.collisionObjectCenters[id] += self.collisionVelocities[id] * self.dt
        
    #dummy transform for default value
    def noTransform(time: ti.f64):
        return -1, -1

    #add half space collision object
    def addHalfSpace(self, center, normal, surface, friction, transform = noTransform):
        
        if not self.symplectic and not surface == self.surfaceSticky:
            raise ValueError("ERROR: Non-sticky boundaries not implemented yet for implicit")

        self.collisionObjectCenters[self.collisionObjectCount] = ti.Vector(list(center)) #save the center so we can update this later
        self.collisionVelocities[self.collisionObjectCount] = ti.Vector([0.0, 0.0]) if self.dim == 2 else ti.Vector([0.0, 0.0, 0.0]) #add a dummy velocity for now
        self.collisionTypes[self.collisionObjectCount] = surface
        self.collisionObjectCount += 1 #update count
        self.transformCallbacks.append(transform) #save the transform callback

        #Code adapted from mpm_solver.py here: https://github.com/taichi-dev/taichi_elements/blob/master/engine/mpm_solver.py
        #center = list(center)
        # normalize normal
        normal_scale = 1.0 / math.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)

        if surface == self.surfaceSticky and friction != 0:
            raise ValueError('friction must be 0 on sticky surfaces.')

        @ti.kernel
        def collide(id: ti.i32):
            for I in ti.grouped(self.grid_m1):
                if self.separable[I] != 1:
                    #treat as one field
                    nodalMass = self.grid_m1[I]
                    if nodalMass > 0:
                        updatedCenter = self.collisionObjectCenters[id]
                        offset = I * self.dx - updatedCenter
                        n = ti.Vector(normal)
                        if offset.dot(n) < 0:

                            #Set boundaries for implicit
                            if not self.symplectic:
                                gid = self.grid_idx[I]
                                self.boundary[gid] = 1 #mark this node as having a boundary condition
                                #continue

                            if self.collisionTypes[id] == self.surfaceSticky:
                                self.grid_v1[I] = self.collisionVelocities[id] #set velocity to be the collision object's velocity
                            else:
                                v = self.grid_v1[I] #divide out the mass to get velocity
                                normal_component = n.dot(v)

                                if self.collisionTypes[id] == self.surfaceSlip:
                                    # Project out all normal component
                                    v = v - n * normal_component
                                else:
                                    # Project out only inward normal component
                                    v = v - n * min(normal_component, 0)

                                if normal_component < 0 and v.norm() > 1e-30:
                                    # apply friction here
                                    v = v.normalized() * max(0, v.norm() + normal_component * friction)

                                self.grid_v1[I] = v + (n * n.dot(self.collisionVelocities[id]))

                else:
                    #treat as two fields
                    nodalMass1 = self.grid_m1[I]
                    nodalMass2 = self.grid_m2[I]
                    if nodalMass1 > 0 and nodalMass2 > 0:
                        updatedCenter = self.collisionObjectCenters[id]
                        offset = I * self.dx - updatedCenter
                        n = ti.Vector(normal)
                        if offset.dot(n) < 0:

                            #Set boundaries for implicit
                            if not self.symplectic:
                                gid = self.grid_idx[I]
                                sgid = self.grid_sep_idx[I]
                                ndof = self.activeNodes[None]
                                self.boundary[gid] = 1 #mark this node as a boundary (field 1)
                                self.boundary[ndof + sgid] = 1 #mark this node as a boundary (field 2)
                                #continue

                            if self.collisionTypes[id] == self.surfaceSticky:
                                self.grid_v1[I] = self.collisionVelocities[id] #set velocity to be the collision object's velocity
                                self.grid_v2[I] = self.collisionVelocities[id] #set velocity to be the collision object's velocity
                            else:
                                v1 = self.grid_v1[I] #divide out the mass to get velocity
                                v2 = self.grid_v2[I] #divide out the mass to get velocity

                                normal_component1 = n.dot(v1)
                                normal_component2 = n.dot(v2)

                                if self.collisionTypes[id] == self.surfaceSlip:
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

                                self.grid_v1[I] = v1 + (n * n.dot(self.collisionVelocities[id]))
                                self.grid_v2[I] = v2 + (n * n.dot(self.collisionVelocities[id]))

        self.collisionCallbacks.append(collide)

    #add spherical collision object
    def addSphereCollider(self, center, radius, surface, transform = noTransform):
        
        if not self.symplectic and not surface == self.surfaceSticky:
            raise ValueError("ERROR: Non-sticky boundaries not implemented yet for implicit")

        self.collisionObjectCenters[self.collisionObjectCount] = ti.Vector(list(center)) #save the center so we can update this later
        self.collisionVelocities[self.collisionObjectCount] = ti.Vector([0.0, 0.0]) if self.dim == 2 else ti.Vector([0.0, 0.0, 0.0]) #add a dummy velocity for now
        self.collisionTypes[self.collisionObjectCount] = surface
        self.collisionObjectCount += 1 #update count
        self.transformCallbacks.append(transform) #save the transform callback

        #Code adapted from mpm_solver.py here: https://github.com/taichi-dev/taichi_elements/blob/master/engine/mpm_solver.py

        @ti.kernel
        def collide(id: ti.i32):
            for I in ti.grouped(self.grid_m1):
                if self.separable[I] != 1:
                    #treat as one field
                    nodalMass = self.grid_m1[I]
                    if nodalMass > 0:
                        updatedCenter = self.collisionObjectCenters[id]
                        offset = I * self.dx - updatedCenter
                        if offset.norm_sqr() < radius * radius:

                            #Set boundaries for implicit
                            if not self.symplectic:
                                gid = self.grid_idx[I]
                                self.boundary[gid] = 1 #mark this node as having a boundary condition
                                #continue

                            if self.collisionTypes[id] == self.surfaceSticky:
                                self.grid_v1[I] = self.collisionVelocities[id] #set velocity to be the collision object's velocity
                            else:
                                v = self.grid_v1[I]
                                normal = offset.normalized(1e-5)
                                normal_component = normal.dot(v)

                                if self.collisionTypes[id] == self.surfaceSlip:
                                    # Project out all normal component
                                    v = v - normal * normal_component
                                else:
                                    # Project out only inward normal component
                                    v = v - normal * min(normal_component, 0)

                                self.grid_v1[I] = v + (normal * normal.dot(self.collisionVelocities[id]))

                else:
                    #treat as two fields
                    nodalMass1 = self.grid_m1[I]
                    nodalMass2 = self.grid_m2[I]
                    if nodalMass1 > 0 and nodalMass2 > 0:
                        updatedCenter = self.collisionObjectCenters[id]
                        offset = I * self.dx - updatedCenter
                        if offset.norm_sqr() < radius * radius:

                            #Set boundaries for implicit
                            if not self.symplectic:
                                gid = self.grid_idx[I]
                                sgid = self.grid_sep_idx[I]
                                ndof = self.activeNodes[None]
                                self.boundary[gid] = 1 #mark this node as a boundary (field 1)
                                self.boundary[ndof + sgid] = 1 #mark this node as a boundary (field 2)
                                #continue

                            if self.collisionTypes[id] == self.surfaceSticky:
                                self.grid_v1[I] = self.collisionVelocities[id] #set velocity to be the collision object's velocity
                                self.grid_v2[I] = self.collisionVelocities[id] #set velocity to be the collision object's velocity
                            else:
                                v1 = self.grid_v1[I] #divide out the mass to get velocity
                                v2 = self.grid_v2[I] #divide out the mass to get velocity
                                normal = offset.normalized()
                                normal_component1 = normal.dot(v1)
                                normal_component2 = normal.dot(v2)

                                if self.collisionTypes[id] == self.surfaceSlip:
                                    # Project out all normal component
                                    v1 = v1 - normal * normal_component1
                                    v2 = v2 - normal * normal_component2
                                else:
                                    # Project out only inward normal component
                                    v1 = v1 - normal * min(normal_component1, 0)
                                    v2 = v2 - normal * min(normal_component2, 0)

                                self.grid_v1[I] = v1 + (normal * normal.dot(self.collisionVelocities[id]))
                                self.grid_v2[I] = v2 + (normal * normal.dot(self.collisionVelocities[id]))

        self.collisionCallbacks.append(collide)

    #----------------Adding External Forces------------------

    def addImpulse(self, center, strength, startTime = 0.0, duration = 1.0):

        self.useImpulse = True
        self.impulseCenter = ti.Vector(list(center))
        self.impulseStrength = strength
        self.impulseStartTime = startTime
        self.impulseDuration = duration
    
    #----------------Damage Models------------------

    def addTimeToFailureDamage(self, damageList, cf, sigmaFRef, vRef, m):

        #Set up parameters from python scope so we can later compute the weibull distribution in the reset kernel (ti.random only usable in kernels)
        self.damageList = np.array(damageList)
        idx = 0
        for i in range(self.numObjects):
            objCount = self.particleCounts[i]
            for j in range(objCount):
                self.useTimeToFailureDamageList[idx] = damageList[i]
                idx += 1 

        self.cf = cf
        self.timeToFail = self.cf / self.l0
        self.sigmaFRef = sigmaFRef
        self.vRef = vRef
        self.m = m
        if self.useDamage:
            raise ValueError('ERROR: you can only use one damage model at a time!')
        else:
            self.useDamage = True

    def addAnisoMPMDamage(self, damageList, eta, dMin, sigmaC = -1.0, percentStretch = -1.0, zeta = 1.0):

        if percentStretch < 0 and sigmaC < 0:
            raise ValueError('ERROR: you must set either percentStretch or sigmaC to use AnisoMPM Damage')
        elif percentStretch > 0 and sigmaC > 0:
            raise ValueError('ERROR: you cannot set both percentStretch and sigmaC')

        print("[AnisoMPM Damage] Simulating with AnisoMPM Damage:")
        if percentStretch > 0: 
            print("[AnisoMPM Damage] Percent Stretch: ", percentStretch)
        else:
            print("[AnisoMPM Damage] SigmaC:", sigmaC)
        print("[AnisoMPM Damage] Eta: ", eta)
        print("[AnisoMPM Damage] Zeta: ", zeta)
        print("[AnisoMPM Damage] dMin: ", dMin)
        self.damageList = np.array(damageList)
        self.sigmaCRef = sigmaC
        self.percentStretch = percentStretch
        self.eta = eta
        self.zeta = zeta
        self.dMin = dMin
        self.minDp = 1.0
        self.useAnisoMPMDamage = True
        if self.useDamage:
            raise ValueError('ERROR: you can only use one damage model at a time!')
        else:
            self.useDamage = True

    def addRankineDamage(self, damageList, Gf, dMin, percentStretch = -1.0, sigmaFRef = -1.0):

        if percentStretch < 0 and sigmaFRef < 0:
            raise ValueError('ERROR: you must set either percentStretch or sigmaFRef to use Rankine Damage')
        elif percentStretch > 0 and sigmaFRef > 0:
            raise ValueError('ERROR: you cannot set both percentStretch and sigmaFRef')

        print("[Rankine Damage] Simulating with Rankine Damage:")
        if percentStretch > 0: 
            print("[Rankine Damage] Percent Stretch: ", percentStretch)
        else:
            print("[Rankine Damage] SigmaFRef:", sigmaFRef)
        print("[Rankine Damage] Gf: ", Gf)
        print("[Rankine Damage] dMin: ", dMin)
        self.damageList = np.array(damageList)
        self.percentStretch = percentStretch
        self.sigmaFRef = sigmaFRef
        self.dMin = dMin
        self.Gf = Gf
        self.minDp = 1.0
        self.useRankineDamage = True
        if self.useDamage:
            raise ValueError('ERROR: you can only use one damage model at a time!')
        else:
            self.useDamage = True

    def addWeibullDistribution(self, vRef, m):

        if self.useRankineDamage == False:
            raise ValueError('ERROR: You must use Rankine with Weibull')

        self.useWeibull = True
        self.vRef = vRef
        self.m = m
        print("[Weibull Distribution] m: ", m)
        if self.useDamage == False:
            raise ValueError('ERROR: you must set a damage model before adding a Weibull distributed sigmaF!')
        
    #------------Simulation Routines---------------

    #Simulation Explicit Substep
    def substepExplicit(self):

        with Timer("Reinitialize Structures"):
            self.grid.deactivate_all() #clear sparse grid structures
            self.grid2.deactivate_all()
            self.reinitializeStructures()

        with Timer("Build Particle IDs"):
            self.build_pid()

        #these routines are unique to DFGMPM
        if self.useDFG:
            #print("A")
            with Timer("Back Grid Sort"):
                self.backGridSort()
                #print("B")
            with Timer("Particle Neighbor Sorting"):
                self.particleNeighborSorting()
                #print("C")
            #Only perform surface detection on the very first substep to prevent artificial DFG fracture
            if self.elapsedTime == 0:
                with Timer("Surface Detection"):
                    self.surfaceDetection()
                    #print("D")
            if self.useRankineDamage:
                with Timer("Update Damage"): #NOTE: make sure to do this before we compute the damage gradients!
                    self.updateDamage()
                    #print("E")
            with Timer("Compute Particle DGs"):
                self.computeParticleDG()
            with Timer("Compute Grid DGs"):
                self.computeGridDG()
            with Timer("Mass P2G"):
                self.massP2G()
            with Timer("Compute Separability"):
                self.computeSeparability()
            if self.useAnisoMPMDamage:
                with Timer("Damage P2G"):
                    self.damageP2G()
                with Timer("Compute Laplacians"):
                    self.computeLaplacians()
                with Timer("Update AnisoMPM Damage"):
                    self.updateAnisoMPMDamage()
        
        with Timer("Momentum P2G and Forces"):
            self.momentumP2GandComputeForces()
        with Timer("Momentum To Velocity"):
            self.momentumToVelocity()
        with Timer("Add Grid Forces"):
            self.addGridForces()
        with Timer("Add Gravity"):
            self.addGravity()

        if self.useImpulse:
            if self.elapsedTime >= self.impulseStartTime and self.elapsedTime < (self.impulseStartTime + self.impulseDuration):
                with Timer("Apply Impulse"):
                    self.applyImpulse()

        if self.useDFG:
            with Timer("Frictional Contact"):
                self.computeContactForces()
            with Timer("Add Contact Forces"):
                self.addContactForces()

        with Timer("Collision Objects"):
            for i in range(self.collisionObjectCount):
                t, v = self.transformCallbacks[i](self.elapsedTime) #get the current translation and velocity based on current time
                if t == -1 and v == -1: #if we used the dummy function, noTransform
                    vel = (0.0, 0.0) if self.dim == 2 else (0.0, 0.0, 0.0)
                self.collisionVelocities[i] = ti.Vector(vel)
                self.updateCollisionObjects(i)
                self.collisionCallbacks[i](i)
        with Timer("G2P"):
            self.G2P()

        self.elapsedTime += self.dt #update elapsed time

    #Simulation Implicit Substep
    def substepImplicit(self):

        with Timer("Reinitialize Structures"):
            self.grid.deactivate_all() #clear sparse grid structures
            self.grid2.deactivate_all()
            self.reinitializeStructures()

        with Timer("Build Particle IDs"):
            self.build_pid()

        #these routines are unique to DFGMPM
        if self.useDFG:
            #print("A")
            with Timer("Back Grid Sort"):
                self.backGridSort()
                #print("B")
            with Timer("Particle Neighbor Sorting"):
                self.particleNeighborSorting()
                #print("C")
            #Only perform surface detection on the very first substep to prevent artificial DFG fracture
            if self.elapsedTime == 0:
                with Timer("Surface Detection"):
                    self.surfaceDetection()
                    #print("D")
            if self.useRankineDamage:
                with Timer("Update Damage"): #NOTE: make sure to do this before we compute the damage gradients!
                    self.updateDamage()
                    #print("E")
            with Timer("Compute Particle DGs"):
                self.computeParticleDG()
            with Timer("Compute Grid DGs"):
                self.computeGridDG()
            with Timer("Mass P2G"):
                self.massP2G()
            with Timer("Compute Separability"):
                self.computeSeparability()
            if self.useAnisoMPMDamage:
                with Timer("Damage P2G"):
                    self.damageP2G()
                with Timer("Compute Laplacians"):
                    self.computeLaplacians()
                with Timer("Update AnisoMPM Damage"):
                    self.updateAnisoMPMDamage()
        
        with Timer("Momentum P2G and Forces"):
            self.momentumP2GandComputeForces()
        with Timer("Momentum To Velocity"):
            self.momentumToVelocity()

        #NOTE: don't add gravity here because we build gravity into our implicit solver

        #NOTE: must be incorporated into the RHS of implicit solve! (like gravity)
        if self.useImpulse:
            if self.elapsedTime >= self.impulseStartTime and self.elapsedTime < (self.impulseStartTime + self.impulseDuration):
                with Timer("Apply Impulse"):
                    self.applyImpulse()

        #Now, after we have momentum transferred and external forces applied (gravity and impulses), we can solve implicitly for the velocity update
        #Iterate Collision Objects to detect which grid nodes have collisions
        with Timer("Collision Objects"):
            self.boundary.fill(0)
            for i in range(self.collisionObjectCount):
                t, v = self.transformCallbacks[i](self.elapsedTime) #get the current translation and velocity based on current time
                if t == -1 and v == -1: #if we used the dummy function, noTransform
                    vel = (0.0, 0.0) if self.dim == 2 else (0.0, 0.0, 0.0)
                self.collisionVelocities[i] = ti.Vector(vel)
                self.updateCollisionObjects(i)
                self.collisionCallbacks[i](i)

        #Here for implicit we compute n_cm1 and n_cm2 for each separable node
        if self.useDFG:
            with Timer("Frictional Contact"):
                self.computeContactForces()

        with Timer("Newton Solve"):
            self.implicitNewton()

        #Frictional Contact, this will be worked into the newton solve directly (eventually)
        # if self.useDFG:
        #     with Timer("Frictional Contact"):
        #         self.computeContactForces()
        #     with Timer("Add Contact Forces"):
        #         self.addContactForces()

        with Timer("G2P"):
            self.G2P()

        self.elapsedTime += self.dt #update elapsed time

    @ti.kernel
    def reset(self, arr: ti.ext_arr(), partCount: ti.ext_arr(), initVel: ti.ext_arr(), pMasses: ti.ext_arr(), pVols: ti.ext_arr(), EList: ti.ext_arr(), nuList: ti.ext_arr(), damageList: ti.ext_arr()):
        self.gravity[None] = ti.Vector.zero(float, self.dim)
        self.gravity[None][1] = self.gravMag
        self.constraintsViolated[None] = 0
        stretchedSigma = 0.0
        for i in range(self.numParticles):
            self.x[i] = [ti.cast(arr[i,0], ti.f64), ti.cast(arr[i,1], ti.f64)] if self.dim == 2 else [ti.cast(arr[i,0], ti.f64), ti.cast(arr[i,1], ti.f64), ti.cast(arr[i,2], ti.f64)] 
            self.material[i] = 0
            self.v[i] = [0, 0] if self.dim == 2 else [0, 0, 0]
            self.F[i] = ti.Matrix.identity(float, self.dim)
            self.Jp[i] = 1
            self.C[i] = ti.Matrix.zero(float, self.dim, self.dim)
            self.Dp[i] = 0
            self.sp[i] = 0
            self.sigmaMax[i] = 0.0
            if self.useAnisoMPMDamage: 
                self.damageLaplacians[i] = 0.0
                self.dTildeH[i] = 0.0
            # if (self.x[i][0] > 0.45) and self.x[i][1] > 0.546 and self.x[i][1] < 0.554: #put damaged particles as a band in the center
            #     self.Dp[i] = 1
            #     self.material[i] = 1
            # if (self.x[i][0] < 0.55) and self.x[i][1] > 0.446 and self.x[i][1] < 0.454: #put damaged particles as a band in the center
            #     self.Dp[i] = 1
            #     self.material[i] = 1
            # if (self.x[i][0] > 0.48 and self.x[i][0] < 0.51):
            #     self.Dp[i] = 1
            #     self.material[i] = 1
        
        #Set different settings for different objects (initVel, mass, volume, and surfaceThreshold for example)
        for serial in range(1):
            idx = 0
            for i in range(self.numObjects):
                objCount = partCount[i]
                for j in range(objCount):
                    self.v[idx] = [ti.cast(initVel[i,0], ti.f64), ti.cast(initVel[i,1], ti.f64)] if self.dim == 2 else [ti.cast(initVel[i,0], ti.f64), ti.cast(initVel[i,1], ti.f64), ti.cast(initVel[i,2], ti.f64)]
                    self.mp[idx] = ti.cast(pMasses[i], ti.f64)
                    self.Vp[idx] = ti.cast(pVols[i], ti.f64)
                    #self.st[idx] = ti.cast(surfaceThresholds[i], ti.f64)
                    E = ti.cast(EList[i], ti.f64)
                    nu = ti.cast(nuList[i], ti.f64)
                    self.mu[idx] = E / (2 * (1 + nu))
                    self.la[idx] = E * nu / ((1+nu) * (1 - 2 * nu))
                    if self.useAnisoMPMDamage:
                        self.useAnisoMPMDamageList[idx] = ti.cast(damageList[i], ti.i32)
                        if self.percentStretch > 0:
                            #compute sigmaC based on percentStretch
                            stretch = 1 + self.percentStretch
                            stretchF = stretch * ti.Matrix.identity(float, self.dim)
                            stretchJ = stretch**self.dim
                            U, sig, V = ti.svd(stretchF)
                            stretchKirchoff = self.kirchoff_FCR(stretchF, U@V.transpose(), stretchJ, self.mu[idx], self.la[idx])
                            e, v1, v2 = self.eigenDecomposition2D(stretchKirchoff / stretchJ) #TODO:3D
                            stretchedSigma = e[0] if e[0] > e[1] else e[1] #TODO:3D
                            self.sigmaC[idx] = stretchedSigma
                        else:
                            self.sigmaC[idx] = self.sigmaCRef
                    if self.useRankineDamage:
                        self.useRankineDamageList[idx] = ti.cast(damageList[i], ti.i32)
                        if self.percentStretch > 0:
                            #compute sigmaF based on percentStretch
                            stretch = 1 + self.percentStretch
                            stretchF = stretch * ti.Matrix.identity(float, self.dim)
                            stretchJ = stretch**self.dim
                            U, sig, V = ti.svd(stretchF)
                            stretchKirchoff = self.kirchoff_FCR(stretchF, U@V.transpose(), stretchJ, self.mu[idx], self.la[idx])
                            e, v1, v2 = self.eigenDecomposition2D(stretchKirchoff / stretchJ) #TODO:3D
                            stretchedSigma = e[0] if e[0] > e[1] else e[1] #TODO:3D
                            self.sigmaF[idx] = stretchedSigma 
                        else:
                            self.sigmaF[idx] = self.sigmaFRef #if we don't have a percentStretch we instead use sigmaFRef (which we should have)
                    idx += 1 

        if self.useRankineDamage:
            if self.percentStretch > 0:
                print("[Rankine Damage] Stretched SigmaF:", stretchedSigma)

        if self.useAnisoMPMDamage:
            if self.percentStretch > 0:
                print("[AnisoMPM Damage] Stretched SigmaC:", stretchedSigma)

        #Now set up damage settings
        #Compute Weibull Distributed SigmaF for TimeToFailure Model
        Hs = 0.0
        for p in range(self.numParticles):
            if self.useWeibull:
                if self.useRankineDamageList[p]:
                    R = ti.cast(ti.random(ti.f32), ti.f64) #ti.random is broken for f64, so use f32
                    sigmaF = stretchedSigma
                    if self.percentStretch <= 0:
                        sigmaF = self.sigmaFRef #use this if we don't have the stretch
                    self.sigmaF[p] = sigmaF * ( ((self.vRef * ti.log(R)) / (self.Vp[p] * ti.log(0.5)))**(1.0 / self.m) )
            #Now compute Hs regardless of whether we use Weibull
            if self.useRankineDamageList[p]:
                G = self.mu[p]
                la = self.la[p]
                E = (G*(3*la + 2*G)) / (la + G) #recompute E for this particle
                #print('reconstructed E: ', E)
                HsBar = (self.sigmaF[p] * self.sigmaF[p]) / (2 * E * self.Gf)
                Hs = (HsBar * self.l0) / (1 - (HsBar * self.l0))
                self.Hs[p] = Hs

        if self.useWeibull == False and self.useRankineDamage:
            print("[Rankine Damage] Hs:", Hs)

    def writeData(self, frame: ti.i32, s: ti.i32):
        
        #Write PLY Files
        #initialize writer and numpy arrays to hold data
        writer = ti.PLYWriter(num_vertices=self.numParticles)
        np_xp = np.zeros((self.numParticles, self.dim), dtype=float)
        np_vp = np.zeros((self.numParticles, self.dim), dtype=float)
        np_DGp = np.zeros((self.numParticles, self.dim), dtype=float)
        np_mp = np.zeros(self.numParticles, dtype=float)
        np_Dp = np.zeros(self.numParticles, dtype=float)
        np_sigmaF = np.zeros(self.numParticles, dtype=float)
        np_sigmaMax = np.zeros(self.numParticles, dtype=float)
        np_dLaplacian = np.zeros(self.numParticles, dtype=float)
        np_sp = np.zeros(self.numParticles, dtype=int)
        np_useDamage = np.zeros(self.numParticles, dtype=int)
        for i in range(self.numParticles):
            for d in ti.static(range(self.dim)):
                np_xp[i,d] = self.x[i][d]
                np_vp[i,d] = self.v[i][d]
                np_DGp[i,d] = self.particleDG[i][d]
            np_mp[i] = self.mp[i]
            np_Dp[i] = self.Dp[i]
            np_sigmaF[i] = self.sigmaF[i]
            np_sigmaMax[i] = self.sigmaMax[i]
            np_dLaplacian[i] = self.damageLaplacians[i]
            np_sp[i] = self.sp[i]
            np_useDamage[i] = self.useAnisoMPMDamageList[i]
        if self.dim == 2: 
            writer.add_vertex_pos(np_xp[:,0], np_xp[:, 1], np.zeros(self.numParticles)) #add position
            writer.add_vertex_channel("vx", "double", np_vp[:,0]) #add vx
            writer.add_vertex_channel("vy", "double", np_vp[:,1]) #add vy
            writer.add_vertex_channel("DGx", "double", np_DGp[:,0]) #add DGx
            writer.add_vertex_channel("DGy", "double", np_DGp[:,1]) #add DGy
        elif self.dim == 3: 
            writer.add_vertex_pos(np_xp[:,0], np_xp[:, 1], np_xp[:,2]) #add position
            writer.add_vertex_channel("vx", "double", np_vp[:,0]) #add vx
            writer.add_vertex_channel("vy", "double", np_vp[:,1]) #add vy
            writer.add_vertex_channel("vz", "double", np_vp[:,2]) #add vz
            writer.add_vertex_channel("DGx", "double", np_DGp[:,0]) #add DGx
            writer.add_vertex_channel("DGy", "double", np_DGp[:,1]) #add DGy
            writer.add_vertex_channel("DGz", "double", np_DGp[:,2]) #add DGz
        writer.add_vertex_channel("m_p", "double", np_mp) #add damage
        writer.add_vertex_channel("Dp", "double", np_Dp) #add damage
        if self.useRankineDamage: writer.add_vertex_channel("sigmaF", "double", np_sigmaF) #add sigmaF
        writer.add_vertex_channel("sigmaMax", "double", np_sigmaMax) #add sigmaMax
        if self.useAnisoMPMDamage: writer.add_vertex_channel("damageLaplacian", "double", np_dLaplacian) #add damageLaplacians
        writer.add_vertex_channel("sp", "int", np_sp) #add surface tracking
        writer.add_vertex_channel("useDamage", "int", np_useDamage)
        if(s == -1):
            writer.export_frame(frame, self.outputPath)
        else:
            writer.export_frame(frame * self.numSubsteps + s, self.outputPath)

        if ti.static(self.dim == 2) and self.activeNodes[None] != 0:
            gridX = np.zeros((self.activeNodes[None], 2), dtype=float) #format as 1d array of nodal positions
            np_separability = np.zeros(self.activeNodes[None], dtype=int)
            gridNormals = np.zeros((self.activeNodes[None], 4), dtype=float)
            gridMasses = np.zeros((self.activeNodes[None], 2), dtype=float)
            gridVelocities = np.zeros((self.activeNodes[None], 4), dtype=float)
            gridForces = np.zeros((self.activeNodes[None], 4), dtype=float)
            gridFrictionForces = np.zeros((self.activeNodes[None], 4), dtype=float)
            np_DG = np.zeros((self.activeNodes[None], 2), dtype=float)
            np_separabilityValue = np.zeros((self.activeNodes[None], 2), dtype=float)
            np_gridDamage = np.zeros((self.activeNodes[None], 2), dtype=float)
            gridIdx = 0
            for i in range(self.activeNodes[None]):
                idx = self.dof2idx[i]
                I = ti.Vector([idx[0], idx[1]])
                i = idx[0]
                j = idx[1]
                gridX[gridIdx,0] = i * self.dx
                gridX[gridIdx,1] = j * self.dx
                np_separability[gridIdx] = self.separable[i,j] #grab separability
                np_DG[gridIdx, 0] = self.gridDG[i, j][0]
                np_DG[gridIdx, 1] = self.gridDG[i, j][1]
                np_separabilityValue[gridIdx, 0] = self.gridSeparability[i, j][0]
                np_separabilityValue[gridIdx, 1] = self.gridSeparability[i, j][1]
                np_gridDamage[gridIdx, 0] = self.grid_d[i, j][0]
                np_gridDamage[gridIdx, 1] = self.grid_d[i, j][1]
                gridVelocities[gridIdx, 0] = self.grid_v1[i, j][0]
                gridVelocities[gridIdx, 1] = self.grid_v1[i, j][1]
                gridVelocities[gridIdx, 2] = self.grid_v2[i, j][0]
                gridVelocities[gridIdx, 3] = self.grid_v2[i, j][1]
                gridForces[gridIdx, 0] = self.grid_f1[i, j][0]
                gridForces[gridIdx, 1] = self.grid_f1[i, j][1]
                gridForces[gridIdx, 2] = self.grid_f2[i, j][0]
                gridForces[gridIdx, 3] = self.grid_f2[i, j][1]
                gridMasses[gridIdx, 0] = self.grid_m1[i, j]
                gridMasses[gridIdx, 1] = self.grid_m2[i, j]
                gridNormals[gridIdx, 0] = self.grid_n1[i, j][0]
                gridNormals[gridIdx, 1] = self.grid_n1[i, j][1]
                gridNormals[gridIdx, 2] = self.grid_n2[i, j][0]
                gridNormals[gridIdx, 3] = self.grid_n2[i, j][1]
                if self.separable[i, j] == 1:
                    gridFrictionForces[gridIdx, 0] = self.grid_fct1[i, j][0]
                    gridFrictionForces[gridIdx, 1] = self.grid_fct1[i, j][1]
                    gridFrictionForces[gridIdx, 2] = self.grid_fct2[i, j][0]
                    gridFrictionForces[gridIdx, 3] = self.grid_fct2[i, j][1]
                gridIdx += 1
            writer2 = ti.PLYWriter(num_vertices=self.activeNodes[None])
            writer2.add_vertex_pos(gridX[:,0], gridX[:, 1], np.zeros(self.activeNodes[None])) #add position
            writer2.add_vertex_channel("sep", "int", np_separability)
            writer2.add_vertex_channel("sep1", "float", np_separabilityValue[:,0])
            writer2.add_vertex_channel("sep2", "float", np_separabilityValue[:,1])
            if self.useAnisoMPMDamage:
                writer2.add_vertex_channel("d1", "float", np_gridDamage[:,0])
                writer2.add_vertex_channel("d2", "float", np_gridDamage[:,1])
            writer2.add_vertex_channel("DGx", "double", np_DG[:,0]) #add particle DG x
            writer2.add_vertex_channel("DGy", "double", np_DG[:,1]) #add particle DG y
            writer2.add_vertex_channel("Ncm_field1_x", "double", gridNormals[:,0]) #add grid_n for field 1 x
            writer2.add_vertex_channel("Ncm_field1_y", "double", gridNormals[:,1]) #add grid_n for field 1 y
            writer2.add_vertex_channel("Ncm_field2_x", "double", gridNormals[:,2]) #add grid_n for field 2 x
            writer2.add_vertex_channel("Ncm_field2_y", "double", gridNormals[:,3]) #add grid_n for field 2 y
            writer2.add_vertex_channel("f_field1_x", "double", gridForces[:,0]) #add grid_fct for field 1 x
            writer2.add_vertex_channel("f_field1_y", "double", gridForces[:,1]) #add grid_fct for field 1 y
            writer2.add_vertex_channel("f_field2_x", "double", gridForces[:,2]) #add grid_fct for field 2 x
            writer2.add_vertex_channel("f_field2_y", "double", gridForces[:,3]) #add grid_fct for field 2 y
            writer2.add_vertex_channel("fct_field1_x", "double", gridFrictionForces[:,0]) #add grid_fct for field 1 x
            writer2.add_vertex_channel("fct_field1_y", "double", gridFrictionForces[:,1]) #add grid_fct for field 1 y
            writer2.add_vertex_channel("fct_field2_x", "double", gridFrictionForces[:,2]) #add grid_fct for field 2 x
            writer2.add_vertex_channel("fct_field2_y", "double", gridFrictionForces[:,3]) #add grid_fct for field 2 y
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
        
        if(s == -1):
            print('#####[Simulation]: Finished writing frame ', frame, '...')
        else:
            print('#####[Simulation]: Finished writing substep ', s, 'of frame ', frame, '...')

    def simulate(self):
        
        #Printing Dialogue
        print("[Simulation] Particle Count: ", self.numParticles)
        print("[Simulation] Grid Dx: ", self.dx)
        if self.symplectic: 
            print("[Simulation] Explicit Time Step: ", self.dt)
            if self.useDFG: print("[Simulation] Explicit DFGMPM")
            if not self.useDFG: print("[Simulation] Explicit Single Field MPM")
        else:
            print("[Simulation] Implicit MaxDt: ", self.maxDt)
            if self.useDFG: print("[Simulation] Implicit DFGMPM")
            if not self.useDFG: print("[Simulation] Implicit Single Field MPM")
        self.reset(self.vertices, self.particleCounts, self.initialVelocity, self.pMasses, self.pVolumes, self.EList, self.nuList, self.damageList) #init
        
        #Time Stepping
        for frame in range(self.endFrame):
            with Timer("Compute Frame"):
                frameFinished = False
                frameTime = 0.0
                currSubstep = 0
                if(self.verbose == False): 
                    with Timer("Visualization"):
                        self.writeData(frame, -1) #NOTE: activate to write frames only
                while not frameFinished:
                    if(self.verbose): 
                        with Timer("Visualization"):
                            self.writeData(frame, currSubstep) #NOTE: activate to write every substep
                    with Timer("Compute Substep"):
                        if not self.symplectic and self.cfl > 0.0:
                            #If implicit, compute dt based on max velocity
                            maxV = self.getMaxVelocity()
                            computedDt = self.dt
                            if maxV > 0.0: 
                                computedDt = self.cfl * self.dx / (2.0 * maxV) #HOT style dynamic dt
                            self.dt = ti.max(self.minDt, ti.min(self.maxDt, computedDt))
                            print("[Implicit] Max Velocity: ", maxV, "Current Dt: ", self.dt)
                        elif self.symplectic:
                            self.dt = self.maxDt #reset dt (even after making it smaller to finish the substep)

                        #Make sure we don't go beyond frameDt
                        if self.frameDt - frameTime < self.dt * 1.001:
                            frameFinished = True
                            self.dt = self.frameDt - frameTime
                        elif self.frameDt - frameTime < 2*self.dt:
                            self.dt = (self.frameDt - frameTime) / 2.0

                        #Simulation substeps
                        if self.symplectic: 
                            self.substepExplicit()
                        else:
                            self.substepImplicit()

                        currSubstep += 1
                        frameTime += self.dt

                        print("[Simulation] Finished computing substep", currSubstep, "of frame", frame+1, "with dt", self.dt)
                        
        Timer_Print()