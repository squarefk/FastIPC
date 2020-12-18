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
    def __init__(self, endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThreshold, useDFG = True, frictionCoefficient = 0.0, verbose = False, useAPIC = False, flipPicRatio = 0.0):
        
        #Simulation Parameters
        self.endFrame = endFrame
        self.fps = fps
        self.frameDt = 1.0 / fps
        self.dt = dt
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
        self.activeNodes = ti.field(dtype=int, shape=())
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
        
        #Neighbor Search Variables - NOTE: if these values are too low, we get data races!!! Try to keep these as high as possible (RIP to ur RAM)
        self.maxNeighbors = 1024 #TODO: figure out how best to set these/is there a way around them? p sure they came from reference code
        self.maxPPC = 1024

        #DFG Parameters
        self.rp = (3*(dx**2))**0.5 if self.dim == 3 else (2*(dx**2))**0.5 #set rp based on dx (this changes if dx != dy)
        self.dMin = 0.25
        self.minDp = 1.0
        self.fricCoeff = frictionCoefficient

        #Grid Fields (Non Sparse)
        # #---General Sim Stuctures
        # self.grid_d = ti.Vector.field(4, dtype=float, shape=(self.nGrid, self.nGrid)) #grid damage structure, need both fields as well as numerator accumulator and denom accumulator
        # self.grid_q1 = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) # grid node momentum, store two vectors at each grid node (for each field)
        # self.grid_q2 = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) # grid node momentum, store two vectors at each grid node (for each field)
        # self.grid_v1 = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) # grid node velocity, store two vectors at each grid node (for each field)
        # self.grid_v2 = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) # grid node velocity, store two vectors at each grid node (for each field)
        # self.grid_vn1 = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) # use this to store grid v_i^n so we can use this for FLIP
        # self.grid_vn2 = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) # use this to store grid v_i^n so we can use this for FLIP
        # self.grid_m1 = ti.field(dtype=float, shape=(self.nGrid, self.nGrid)) # grid node field 1 mass is nGrid x nGrid, each grid node has a mass for each field
        # self.grid_m2 = ti.field(dtype=float, shape=(self.nGrid, self.nGrid)) # grid node field 2 mass is nGrid x nGrid, each grid node has a mass for each field
        # self.grid_n1 = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) # grid node normals for two field nodes, store two vectors at each grid node (one for each field)
        # self.grid_n2 = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) # grid node normals for two field nodes, store two vectors at each grid node (one for each field)
        # self.grid_f1 = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) # grid node forces so we can apply them later
        # self.grid_f2 = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) # grid node forces so we can apply them later
        # self.grid_fct1 = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) # grid node contact forces for two field nodes, store two vectors at each grid node (one for each field)
        # self.grid_fct2 = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) # grid node contact forces for two field nodes, store two vectors at each grid node (one for each field)
        # #---Active Node Indexing
        # self.grid_idx = ti.field(dtype=int, shape=(self.nGrid,self.nGrid))
        # #---DFG Grid Structures
        # self.gridDG = ti.Vector.field(self.dim, dtype=float, shape=(self.nGrid, self.nGrid)) #grid node damage gradients
        # self.gridMaxNorm = ti.field(dtype=float, shape=(self.nGrid, self.nGrid)) #grid max norm holds the maximum DG norm found for this grid node, this will later help to determine the grid DG
        # self.gridSeparability = ti.Vector.field(4, dtype=float, shape=(self.nGrid, self.nGrid)) # grid separability is nGrid x nGrid x 4, each grid node has a seperability condition for each field and we need to add up the numerator and denominator
        # self.gridMaxDamage = ti.Vector.field(2, dtype=float, shape=(self.nGrid, self.nGrid)) # grid max damage is nGrid x nGrid x 2, each grid node has a max damage from each field
        # self.separable = ti.field(dtype=int, shape=(self.nGrid,self.nGrid)) # whether grid node is separable or not
        
        #Particle Fields
        # #---Lame Parameters
        # self.mu = ti.field(dtype=float, shape=self.numParticles)
        # self.la = ti.field(dtype=float, shape=self.numParticles)
        # #---General Simulation
        # self.x = ti.Vector.field(self.dim, dtype=float, shape=self.numParticles) # position
        # self.v = ti.Vector.field(self.dim, dtype=float, shape=self.numParticles) # velocity
        # self.C = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=self.numParticles) # affine velocity field
        # self.F = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=self.numParticles) # deformation gradient
        # self.material = ti.field(dtype=int, shape=self.numParticles) # material id
        # self.mp = ti.field(dtype=float, shape=self.numParticles) # particle masses
        # self.Vp = ti.field(dtype=float, shape=self.numParticles) # particle volumes
        # self.Jp = ti.field(dtype=float, shape=self.numParticles) # plastic deformation
        # #---DFG Fields
        # self.Dp = ti.field(dtype=float, shape=self.numParticles) #particle damage
        # self.sp = ti.field(dtype=int, shape=self.numParticles) #particle surface boolean int -- 1 = surface particle, 0 = interior particle
        # self.particleDG = ti.Vector.field(self.dim, dtype=float, shape=self.numParticles) #keep track of particle damage gradients 
        # #self.st = ti.field(dtype=float, shape=self.numParticles) #now we can have different thresholds for different objects and particle distributions!        
        # #---AnisoMPM Damage
        # self.sigmaC = ti.field(dtype=float, shape=self.numParticles) #each particle can have different sigmaC
        # self.dTildeH = ti.field(dtype=float, shape=self.numParticles) #keep track of the maximum driving force seen by this particle
        # self.damageLaplacians = ti.field(dtype=float, shape=self.numParticles) #store the damage laplacians
        # self.useAnisoMPMDamageList = ti.field(dtype=int, shape=self.numParticles)
        # #---Rankine Damage
        # self.Hs = ti.field(dtype=float, shape=self.numParticles) #weibull will make these different from eahc other
        # self.sigmaF = ti.field(dtype=float, shape=self.numParticles) #each particle can have different sigmaF based on Weibull dist
        # self.sigmaMax = ti.field(dtype=float, shape=self.numParticles) #track sigmaMax for each particle to visualize stress
        # self.useRankineDamageList = ti.field(dtype=int, shape=self.numParticles)
        # #---Time To Failure Damage
        # self.useTimeToFailureDamageList = ti.field(dtype=int, shape=self.numParticles)               
        
        #Active Fields
        self.particleAF = ti.Vector.field(3**self.dim, dtype=int, shape=self.numParticles) #store which activefield each particle belongs to for the 9 grid nodes it maps to

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

        #Dynamic Particle Structures
        #---Params
        self.max_num_particles = 2**27
        #---Lame Parameters
        self.mu = ti.field(dtype=float)
        self.la = ti.field(dtype=float)
        #---General Simulation
        self.x = ti.Vector.field(self.dim, dtype=float) # position
        self.v = ti.Vector.field(self.dim, dtype=float) # velocity
        self.C = ti.Matrix.field(self.dim, self.dim, dtype=float) # affine velocity field
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=float) # deformation gradient
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
        #---Shape and Then Place Structures
        self.particle = ti.root.dynamic(ti.i, self.max_num_particles, 2**19) #2**20 causes problems in CUDA (maybe asking for too much space)
        self.particle.place(self.mu, self.la, self.x, self.v, self.C, self.F, self.material, self.mp, self.Vp, self.Jp, self.Dp, self.sp, self.particleDG, self.sigmaC, self.dTildeH, self.damageLaplacians, self.useAnisoMPMDamageList, self.Hs, self.sigmaF, self.sigmaMax, self.useRankineDamageList, self.useTimeToFailureDamageList)

        #Sparse Grids
        #---Params
        self.grid_size = 4096
        self.grid_block_size = 128
        self.leaf_block_size = 16 if self.dim == 2 else 8
        self.indices = ti.ij if self.dim == 2 else ti.ijk
        #self.offset = tuple(-self.grid_size // 2 for _ in range(self.dim))
        self.offset = tuple(0 for _ in range(self.dim))
        #---Grid Shapes for PID
        self.grid = ti.root.pointer(self.indices, self.grid_size // self.grid_block_size) # 32
        self.block = self.grid.pointer(self.indices, self.grid_block_size // self.leaf_block_size) # 8
        self.pid = ti.field(int)
        self.block.dynamic(ti.indices(self.dim), 1024 * 1024, chunk_size=self.leaf_block_size**self.dim * 8).place(self.pid, offset=self.offset + (0, ))
        #---Neighbor Search Fields
        #self.gridNumParticles_sparse = ti.field(int)
        #self.backGrid_sparse = ti.field(int)
        #block_component(self.gridNumParticles_sparse) #NOTE: gridNumParticles is nGrid x nGrid
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
        self.grid_f1 = ti.Vector.field(self.dim, dtype=float)
        self.grid_f2 = ti.Vector.field(self.dim, dtype=float)
        self.grid_fct1 = ti.Vector.field(self.dim, dtype=float)
        self.grid_fct2 = ti.Vector.field(self.dim, dtype=float)
        #---DOF Tracking
        self.grid_idx = ti.field(dtype=int)
        #---DFG Grid Structures
        self.gridDG = ti.Vector.field(self.dim, dtype=float)
        self.gridMaxNorm = ti.field(dtype=float)
        self.gridSeparability = ti.Vector.field(4, dtype=float) #numerator field 1, numerator field 2, denom field 1, denom field 2
        self.gridMaxDamage = ti.Vector.field(2, dtype=float) #max damage field 1, max damage field 2
        self.separable = ti.field(dtype=int)
        #---Place on Sparse Grid
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
        block_component(self.grid_idx) #hold a mapping from active grid DOF indeces -> the DOF index
        for dg in self.gridDG.entries: #grid node damage gradients
            block_component(dg)
        block_component(self.gridMaxNorm)  #grid max norm holds the maximum DG norm found for this grid node, this will later help to determine the grid DG
        for sep in self.gridSeparability.entries: # grid separability is nGrid x nGrid x 4, each grid node has a seperability condition for each field and we need to add up the numerator and denominator
            block_component(sep)
        for maxD in self.gridMaxDamage.entries: # grid max damage is nGrid x nGrid x 2, each grid node has a max damage from each field
            block_component(maxD)
        block_component(self.separable) # whether grid node is separable or not

        #TODO: 8 n x n dense grid structures
        #TODO: 6 n x n x 2 grid structures to split into two dense structures
        #TODO: 1 n x n x maxPPC structure to turn into dynamic

        #Sparse Matrix Fields
        MAX_LINEAR = 5000000
        self.dof2idx = ti.Vector.field(self.dim, ti.i32, shape=MAX_LINEAR)

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
    def isInGrid(self, x):
        return 0 <= x[0] and x[0] < self.nGrid and 0 <= x[1] and x[1] < self.nGrid

    ##########

    #Active Field Indexing
    @ti.func
    def activeFieldIndex(self, offset):
        idx = 0
        for d in ti.static(range(self.dim)):
            if d == self.dim - 1:
                #on final index, just add
                idx += offset[d]
            else:
                idx += offset[d]*3
        return idx

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

    #AnisoMPM Function Evaluations

    @ti.func
    def macaulay(self, x):
        return (x + abs(x)) / 2.0

    ##########

    #Simulation Routines
    @ti.kernel
    def reinitializeStructures(self):
        #re-initialize grid quantities
        # for I in ti.grouped(self.grid_m1):
        #     self.grid_q1[I] = [0, 0] #field 1 momentum
        #     self.grid_q2[I] = [0, 0] #field 2 momentum
        #     self.grid_v1[I] = [0, 0] #field 1 vel
        #     self.grid_v2[I] = [0, 0] #field 2 vel
        #     self.grid_vn1[I] = [0, 0] #field 1 vel v_i^n
        #     self.grid_vn2[I] = [0, 0] #field 2 vel v_i^n
        #     self.grid_n1[I] = [0, 0] #field 1 normal
        #     self.grid_n2[I] = [0, 0] #field 2 normal
        #     self.grid_f1[I] = [0, 0] #f1 nodal force
        #     self.grid_f2[I] = [0, 0] #f2 nodal force
        #     self.grid_fct1[I] = [0, 0] #f1 nodal force
        #     self.grid_fct2[I] = [0, 0] #f2 nodal force
        #     self.grid_m1[I] = 0 # field 1 mass
        #     self.grid_m2[I] = 0 #field 2 mass
        #     self.gridSeparability[I] = [0, 0, 0, 0] #stackt fields, and we use the space to add up the numerator and denom for each field
        #     self.gridMaxDamage[I] = [0, 0] #stackt fields
        #     self.gridDG[I] = [0, 0] #reset grid node damage gradients
        #     self.gridMaxNorm[I] = 0 #reset max norm DG found at the grid node
        #     self.separable[I] = -1 #-1 for only one field, 0 for not separable, and 1 for separable
        #     self.grid_d[I] = [0, 0, 0, 0]
        #     self.grid_idx[I] = -1
        
        #Clear neighbor look up structures
        for I in ti.grouped(self.gridNumParticles):
            self.gridNumParticles[I] = 0
        for I in ti.grouped(self.particleNeighbors):
            self.particleNeighbors[I] = -1
        for I in ti.grouped(self.particleAF):
            self.particleAF[I] = [-1, -1, -1, -1, -1, -1, -1, -1, -1] #TODO: 3d
    
    @ti.kernel
    def backGridSort(self):
        #Sort particles into backGrid
        ti.block_dim(256)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            cell = self.backGridIdx(self.x[p]) #grab cell idx (vector of ints)
            offs = ti.atomic_add(self.gridNumParticles[cell], 1) #atomically add one to our grid cell's particle count NOTE: returns the OLD value before add
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
                kirchoff = self.kirchoff_NeoHookean(self.F[p], J, self.mu[p], self.la[p]) #compute kirchoff stress using the NH model from homel2016                
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
                kirchoff = self.kirchoff_NeoHookean(self.F[p], J, self.mu[p], self.la[p]) #compute kirchoff stress using the NH model from homel2016                
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
            self.particleDG[p] = nablaDBar #store particle DG

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

                maxD = max(self.Dp[p], self.sp[p]) #use max of damage and surface particle markers so we detect green case correctly

                #Set Active Fields for each grid node! 
                #afIdx = offset[0]*3 + offset[1] 
                # if ti.static(self.dim == 3): 
                #     afIdx = offset[0]*3 + offset[1]*3 + offset[2]
                if self.particleDG[p].dot(self.gridDG[gridIdx]) >= 0:
                    self.grid_m1[gridIdx] += weight * self.mp[p] #add mass to active field for this particle
                    self.gridSeparability[gridIdx][0] += weight * maxD * self.mp[p] #numerator, field 1
                    self.gridSeparability[gridIdx][2] += weight * self.mp[p] #denom, field 1
                    self.particleAF[p][offset[0]*3 + offset[1]] = 0 #set this particle's AF to 0 for this grid node #TODO: 3D
                    ti.atomic_max(self.gridMaxDamage[gridIdx][0], maxD) #compute the max damage seen in this field at this grid node
                else:
                    self.grid_m2[gridIdx] += weight * self.mp[p] #add mass to active field for this particle
                    self.gridSeparability[gridIdx][1] += weight * maxD * self.mp[p] #numerator, field 2
                    self.gridSeparability[gridIdx][3] += weight * self.mp[p] #denom, field 2
                    self.particleAF[p][offset[0]*3 + offset[1]] = 1 #set this particle's AF to 1 for this grid node #TODO: 3D
                    ti.atomic_max(self.gridMaxDamage[gridIdx][1], maxD) #compute the max damage seen in this field at this grid node

    @ti.kernel
    def computeSeparability(self):
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
                weight = 1.0 #w[i][0] * w[j][1]
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                #Treat as either single-field or two-field
                if self.separable[gridIdx] != 1:
                    #Single Field
                    self.grid_d[gridIdx][0] += weight * self.Dp[p]
                    self.grid_d[gridIdx][2] += weight
                else:
                    #Two-Field
                    #afIdx = offset[0]*3 + offset[1] if self.dim == 2 else offset[0]*3 + offset[1]*3 + offset[2]
                    fieldIdx = self.particleAF[p][offset[0]*3 + offset[1]] #TODO: 3D
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
                    #afIdx = offset[0]*3 + offset[1] if self.dim == 2 else offset[0]*3 + offset[1]*3 + offset[2]
                    fieldIdx = self.particleAF[p][offset[0]*3 + offset[1]] #grab the field that this particle is in for this node #TODO: 3D
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
                kirchoff = self.kirchoff_NeoHookean(self.F[p], J, self.mu[p], self.la[p]) #compute kirchoff stress using the NH model from homel2016                
                e, v1, v2 = self.eigenDecomposition2D(kirchoff / J) #use my eigendecomposition, comes out as three 2D vectors
                sigmaPlus = (self.macaulay(e[0]) * v1.outer_product(v1)) + (self.macaulay(e[1]) * v2.outer_product(v2))

                #Compute Phi
                A = ti.Matrix.identity(float, self.dim) #set up structural tensor for later use
                Asig = A @ sigmaPlus
                sigA = sigmaPlus @ A
                contraction = 0.0
                for i in ti.static(range(self.dim)):
                    for j in ti.static(range(self.dim)):
                        contraction += Asig[i, j] * sigA[i, j] #TODO: 3D
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
            #print('massP2G base before:', base)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)
            #print('massP2G base after:', base)
            #for particle p, compute base index
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
            #kirchoff = self.kirchoff_FCR(self.F[p], U@V.transpose(), J, self.mu[p], self.la[p])
            kirchoff = self.kirchoff_NeoHookean(self.F[p], J, self.mu[p], self.la[p])

            #NOTE Grab the sigmaMax here so we can learn how to better threshold the stress for damage
            e, v1, v2 = self.eigenDecomposition2D(kirchoff / J) #use my eigendecomposition, comes out as three 2D vectors
            self.sigmaMax[p] = e[0] if e[0] > e[1] else e[1] #TODO: 3d

            #P2G for velocity, force update, and update velocity
            for offset in ti.static(ti.grouped(self.stencil_range())): # Loop over grid node stencil
                gridIdx = base + offset
                #print('base:', base)
                #print('gridIDx:', gridIdx)
                dpos = (offset.cast(float) - fx) * self.dx
                weight = 1.0 #w[i][0] * w[j][1] in 2D
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                
                dweight = ti.Vector.zero(float,self.dim)
                if ti.static(self.dim == 2):
                    dweight[0] = self.invDx * dw[offset[0]][0] * w[offset[1]][1]
                    dweight[1] = self.invDx * w[offset[0]][0] * dw[offset[1]][1]
                else:
                    dweight[0] = dw[offset[0]][0]*w[offset[1]][1]*w[offset[2]][2] * self.inv_dx
                    dweight[1] = w[offset[0]][0]*dw[offset[1]][1]*w[offset[2]][2] * self.inv_dx
                    dweight[2] = w[offset[0]][0]*w[offset[1]][1]*dw[offset[2]][2] * self.inv_dx
                
                force = -self.Vp[p] * kirchoff @ dweight

                if self.separable[gridIdx] != 1 or self.useDFG == False: 
                    #print("getting to if case")
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

                elif self.separable[gridIdx] == 1:
                    #treat node as having two fields
                    #afIdx = offset[0]*3 + offset[1] if self.dim == 2 else offset[0]*3 + offset[1]*3 + offset[2]
                    fieldIdx = self.particleAF[p][offset[0]*3 + offset[1]] #grab the field that this particle is in for this node #TODO: 3D
                    if fieldIdx == 0:
                        #field 1
                        if self.useAPIC:
                            self.grid_q1[gridIdx] += self.mp[p] * weight * (self.v[p] + self.C[p] @ dpos) #momentum transfer (APIC)
                        else:
                            self.grid_q1[gridIdx] += self.mp[p] * weight * self.v[p] #momentum transfer (PIC)
                        self.grid_f1[gridIdx] += force                    
                        self.grid_n1[gridIdx] += dweight * self.mp[p] #add to the normal for this field at this grid node, remember we need to normalize it later!
                    elif fieldIdx == 1:
                        #field 2
                        if self.useAPIC:
                            self.grid_q2[gridIdx] += self.mp[p] * weight * (self.v[p] + self.C[p] @ dpos) #momentum transfer (APIC)
                        else:
                            self.grid_q2[gridIdx] += self.mp[p] * weight * self.v[p] #momentum transfer (PIC)
                        self.grid_f2[gridIdx] += force                    
                        self.grid_n2[gridIdx] += dweight * self.mp[p] #add to the normal for this field at this grid node, remember we need to normalize it later!
                    else:
                        print("ERROR: why did we get here???")
                        #raise ValueError('ERROR: invalid field idx somehow')
        #print('particle count for p2g:', particleCount)

    @ti.kernel
    def momentumToVelocity(self):
        self.activeNodes[None] = 0
        for I in ti.grouped(self.grid_m1):
            if self.grid_m1[I] > 0:
                #print("I:", I, " m1:", self.grid_m1[I])
                self.grid_v1[I] = self.grid_q1[I] / self.grid_m1[I]
                self.grid_vn1[I] = self.grid_q1[I] / self.grid_m1[I]

                #Setup our grid indeces <-> DOF mapping
                idx = self.activeNodes[None].atomic_add(1)
                self.grid_idx[I] = idx
                self.dof2idx[idx] = I

            if self.separable[I] == 1:
                self.grid_v2[I] = self.grid_q2[I] / self.grid_m2[I] 
                self.grid_vn2[I] = self.grid_q2[I] / self.grid_m2[I]
        #print('activeNodes:', self.activeNodes[None])

    @ti.kernel
    def addGridForces(self):
        for I in ti.grouped(self.grid_m1):
            if self.grid_m1[I] > 0:
                self.grid_v1[I] += (self.grid_f1[I] * self.dt) / self.grid_m1[I]
            if self.separable[I] == 1:
                self.grid_v2[I] += (self.grid_f2[I] * self.dt) / self.grid_m2[I]

    @ti.kernel
    def addGravity(self):
         #Add Gravity
        for I in ti.grouped(self.grid_m1):
            if self.grid_m1[I] > 0:
                self.grid_v1[I] += self.dt * self.gravity[None]
            if self.separable[I] == 1:
                self.grid_v2[I] += self.dt * self.gravity[None]

    @ti.kernel
    def applyImpulse(self):
        for I in ti.grouped(self.grid_m1):
            dist = self.impulseCenter - (self.dx * I)
            dv = dist / (0.01 + dist.norm()) * self.impulseStrength * self.dt
            if self.grid_m1[I] > 0:
                self.grid_v1[I] += dv
            if self.separable[I] == 1:
                self.grid_v2[I] += dv

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
                if(self.useFrictionalContact):
                    self.grid_v1[I] += (self.grid_fct1[I] / self.grid_m1[I]) * self.dt # use field 1 force to update field 1 particles (for green nodes, ie separable contact)
                    self.grid_v2[I] += (self.grid_fct2[I] / self.grid_m2[I]) * self.dt # use field 2 force to update field 2 particles

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
                weight = 1.0 #w[i][0] * w[j][1] in 2D
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                dweight = ti.Vector.zero(float,self.dim)
                if ti.static(self.dim == 2):
                    dweight[0] = self.invDx * dw[offset[0]][0] * w[offset[1]][1]
                    dweight[1] = self.invDx * w[offset[0]][0] * dw[offset[1]][1]
                else:
                    dweight[0] = dw[offset[0]][0]*w[offset[1]][1]*w[offset[2]][2] * self.inv_dx
                    dweight[1] = w[offset[0]][0]*dw[offset[1]][1]*w[offset[2]][2] * self.inv_dx
                    dweight[2] = w[offset[0]][0]*w[offset[1]][1]*dw[offset[2]][2] * self.inv_dx

                if self.separable[gridIdx] != 1:
                    #treat as one field
                    new_v_PIC += weight * g_v_np1
                    new_v_FLIP += weight * (g_v_np1 - g_v_n)
                    new_C += 4 * self.invDx * weight * g_v_np1.outer_product(dpos)
                    new_F += g_v_np1.outer_product(dweight)
                else:
                    #node has two fields so choose the correct velocity contribution from the node
                    #afIdx = offset[0]*3 + offset[1] if self.dim == 2 else offset[0]*3 + offset[1]*3 + offset[2]
                    fieldIdx = self.particleAF[p][offset[0]*3 + offset[1]] #grab the field that this particle is in for this node #TODO: 3D
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

    #update collision object centers based on the translation and velocity
    @ti.kernel
    def updateCollisionObjects(self, id: ti.i32):
        self.collisionObjectCenters[id] += self.collisionVelocities[id] * self.dt
        
    #dummy transform for default value
    def noTransform(time: ti.f64):
        return (0.0, 0.0), (0.0, 0.0)

    #add half space collision object
    def addHalfSpace(self, center, normal, surface, friction, transform = noTransform):
        
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

    #Simulation substep
    def substep(self):

        with Timer("Reinitialize Structures"):
            self.grid.deactivate_all() #clear sparse grid structures
            self.reinitializeStructures()

        with Timer("Build Particle IDs"):
            self.build_pid()

        #these routines are unique to DFGMPM
        if self.useDFG:
            with Timer("Back Grid Sort"):
                self.backGridSort()
            with Timer("Particle Neighbor Sorting"):
                self.particleNeighborSorting()
            #Only perform surface detection on the very first substep to prevent artificial DFG fracture
            if self.elapsedTime == 0:
                with Timer("Surface Detection"):
                    self.surfaceDetection()
            if self.useRankineDamage:
                with Timer("Update Damage"): #NOTE: make sure to do this before we compute the damage gradients!
                    self.updateDamage()
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
                self.collisionVelocities[i] = ti.Vector(v)
                self.updateCollisionObjects(i)
                self.collisionCallbacks[i](i)
        with Timer("G2P"):
            self.G2P()

        self.elapsedTime += self.dt #update elapsed time

    @ti.kernel
    def reset(self, arr: ti.ext_arr(), partCount: ti.ext_arr(), initVel: ti.ext_arr(), pMasses: ti.ext_arr(), pVols: ti.ext_arr(), EList: ti.ext_arr(), nuList: ti.ext_arr(), damageList: ti.ext_arr()):
        self.gravity[None] = [0, self.gravMag] if self.dim == 2 else [0, self.gravMag, 0]
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
                            stretchKirchoff = self.kirchoff_NeoHookean(stretchF, stretchJ, self.mu[idx], self.la[idx])
                            e, v1, v2 = self.eigenDecomposition2D(stretchKirchoff / stretchJ) #TODO: 3D
                            stretchedSigma = e[0] if e[0] > e[1] else e[1] #TODO: 3D
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
                            stretchKirchoff = self.kirchoff_NeoHookean(stretchF, stretchJ, self.mu[idx], self.la[idx])
                            e, v1, v2 = self.eigenDecomposition2D(stretchKirchoff / stretchJ) #TODO: 3D
                            stretchedSigma = e[0] if e[0] > e[1] else e[1] #TODO: 3D
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
        
        if(s == -1):
            print('[Simulation]: Writing frame ', frame, '...')
        else:
            print('[Simulation]: Writing substep ', s, 'of frame ', frame, '...')

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
            writer2.add_vertex_channel("N_field1_x", "double", gridNormals[:,0]) #add grid_n for field 1 x
            writer2.add_vertex_channel("N_field1_y", "double", gridNormals[:,1]) #add grid_n for field 1 y
            writer2.add_vertex_channel("N_field2_x", "double", gridNormals[:,2]) #add grid_n for field 2 x
            writer2.add_vertex_channel("N_field2_y", "double", gridNormals[:,3]) #add grid_n for field 2 y
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
            
            #OLD WAY (w/o sparse grids)
            #Construct positions for grid nodes
            # gridX = np.zeros((self.nGrid**2, 2), dtype=float) #format as 1d array of nodal positions
            # np_separability = np.zeros(self.nGrid**2, dtype=int)
            # gridNormals = np.zeros((self.nGrid**2, 4), dtype=float)
            # gridMasses = np.zeros((self.nGrid**2, 2), dtype=float)
            # gridVelocities = np.zeros((self.nGrid**2, 4), dtype=float)
            # gridForces = np.zeros((self.nGrid**2, 4), dtype=float)
            # gridFrictionForces = np.zeros((self.nGrid**2, 4), dtype=float)
            # np_DG = np.zeros((self.nGrid**2, 2), dtype=float)
            # np_separabilityValue = np.zeros((self.nGrid**2, 2), dtype=float)
            # np_gridDamage = np.zeros((self.nGrid**2, 2), dtype=float)
            # for i in range(self.nGrid):
            #     for j in range(self.nGrid):
            #         gridIdx = i * self.nGrid + j
            #         gridX[gridIdx,0] = i * self.dx
            #         gridX[gridIdx,1] = j * self.dx
            #         np_separability[gridIdx] = self.separable[i, j] #grab separability
            #         #TODO: 3D and sparse!!!
            #         np_DG[gridIdx, 0] = self.gridDG[i, j][0]
            #         np_DG[gridIdx, 1] = self.gridDG[i, j][1]
            #         np_separabilityValue[gridIdx, 0] = self.gridSeparability[i, j][0]
            #         np_separabilityValue[gridIdx, 1] = self.gridSeparability[i, j][1]
            #         np_gridDamage[gridIdx, 0] = self.grid_d[i, j][0]
            #         np_gridDamage[gridIdx, 1] = self.grid_d[i, j][1]
            #         gridVelocities[gridIdx, 0] = self.grid_v1[i, j][0]
            #         gridVelocities[gridIdx, 1] = self.grid_v1[i, j][1]
            #         gridVelocities[gridIdx, 2] = self.grid_v2[i, j][0]
            #         gridVelocities[gridIdx, 3] = self.grid_v2[i, j][1]
            #         gridForces[gridIdx, 0] = self.grid_f1[i, j][0]
            #         gridForces[gridIdx, 1] = self.grid_f1[i, j][1]
            #         gridForces[gridIdx, 2] = self.grid_f2[i, j][0]
            #         gridForces[gridIdx, 3] = self.grid_f2[i, j][1]
            #         gridMasses[gridIdx, 0] = self.grid_m1[i, j]
            #         gridMasses[gridIdx, 1] = self.grid_m2[i, j]
            #         gridNormals[gridIdx, 0] = self.grid_n1[i, j][0]
            #         gridNormals[gridIdx, 1] = self.grid_n1[i, j][1]
            #         gridNormals[gridIdx, 2] = self.grid_n2[i, j][0]
            #         gridNormals[gridIdx, 3] = self.grid_n2[i, j][1]
            #         if self.separable[i, j] == 1:
            #             gridFrictionForces[gridIdx, 0] = self.grid_fct1[i, j][0]
            #             gridFrictionForces[gridIdx, 1] = self.grid_fct1[i, j][1]
            #             gridFrictionForces[gridIdx, 2] = self.grid_fct2[i, j][0]
            #             gridFrictionForces[gridIdx, 3] = self.grid_fct2[i, j][1]
            # writer2 = ti.PLYWriter(num_vertices=self.nGrid**2)
            # writer2.add_vertex_pos(gridX[:,0], gridX[:, 1], np.zeros(self.nGrid**2)) #add position
            # writer2.add_vertex_channel("sep", "int", np_separability)
            # writer2.add_vertex_channel("sep1", "float", np_separabilityValue[:,0])
            # writer2.add_vertex_channel("sep2", "float", np_separabilityValue[:,1])
            # if self.useAnisoMPMDamage:
            #     writer2.add_vertex_channel("d1", "float", np_gridDamage[:,0])
            #     writer2.add_vertex_channel("d2", "float", np_gridDamage[:,1])
            # writer2.add_vertex_channel("DGx", "double", np_DG[:,0]) #add particle DG x
            # writer2.add_vertex_channel("DGy", "double", np_DG[:,1]) #add particle DG y
            # writer2.add_vertex_channel("N_field1_x", "double", gridNormals[:,0]) #add grid_n for field 1 x
            # writer2.add_vertex_channel("N_field1_y", "double", gridNormals[:,1]) #add grid_n for field 1 y
            # writer2.add_vertex_channel("N_field2_x", "double", gridNormals[:,2]) #add grid_n for field 2 x
            # writer2.add_vertex_channel("N_field2_y", "double", gridNormals[:,3]) #add grid_n for field 2 y
            # writer2.add_vertex_channel("f_field1_x", "double", gridForces[:,0]) #add grid_fct for field 1 x
            # writer2.add_vertex_channel("f_field1_y", "double", gridForces[:,1]) #add grid_fct for field 1 y
            # writer2.add_vertex_channel("f_field2_x", "double", gridForces[:,2]) #add grid_fct for field 2 x
            # writer2.add_vertex_channel("f_field2_y", "double", gridForces[:,3]) #add grid_fct for field 2 y
            # writer2.add_vertex_channel("fct_field1_x", "double", gridFrictionForces[:,0]) #add grid_fct for field 1 x
            # writer2.add_vertex_channel("fct_field1_y", "double", gridFrictionForces[:,1]) #add grid_fct for field 1 y
            # writer2.add_vertex_channel("fct_field2_x", "double", gridFrictionForces[:,2]) #add grid_fct for field 2 x
            # writer2.add_vertex_channel("fct_field2_y", "double", gridFrictionForces[:,3]) #add grid_fct for field 2 y
            # writer2.add_vertex_channel("v_field1_x", "double", gridVelocities[:,0])
            # writer2.add_vertex_channel("v_field1_y", "double", gridVelocities[:,1])
            # writer2.add_vertex_channel("v_field2_x", "double", gridVelocities[:,2])
            # writer2.add_vertex_channel("v_field2_y", "double", gridVelocities[:,3])
            # writer2.add_vertex_channel("m1", "double", gridMasses[:,0])
            # writer2.add_vertex_channel("m2", "double", gridMasses[:,1])

    def simulate(self):
        print("[Simulation] Particle Count: ", self.numParticles)
        print("[Simulation] Grid Dx: ", self.dx)
        print("[Simulation] Time Step: ", self.dt)
        self.reset(self.vertices, self.particleCounts, self.initialVelocity, self.pMasses, self.pVolumes, self.EList, self.nuList, self.damageList) #init
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