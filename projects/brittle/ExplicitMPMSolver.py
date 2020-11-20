import taichi as ti
import numpy as np
import math
from common.utils.timer import *

@ti.data_oriented
class ExplicitMPMSolver:
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
    def __init__(self, endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, verbose = False, useAPIC = False, flipPicRatio = 0.0):
        
        #Simulation Parameters
        self.endFrame = endFrame
        self.fps = fps
        self.frameDt = 1.0 / fps
        self.dt = dt
        self.elapsedTime = 0.0 #keep a concept of elapsed time
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
        self.EList = np.array(EList)
        self.nuList = np.array(nuList)
        self.dx = dx
        self.inv_dx = 1.0 / dx
        self.nGrid = ti.ceil(self.inv_dx)
        self.mu = ti.field(dtype=float, shape=self.numParticles)
        self.la = ti.field(dtype=float, shape=self.numParticles)
        self.gravMag = gravity
        self.verbose = verbose
        self.useAPIC = useAPIC
        self.flipPicRatio = flipPicRatio #default to 0 which means full PIC
        if flipPicRatio < 0.0 or flipPicRatio > 1.0:
            raise ValueError('flipPicRatio must be between 0 and 1')
        
        #Collision Variables
        self.collisionCallbacks = [] #hold function callbacks for post processing velocity
        self.transformCallbacks = [] #hold callbacks for moving our boundaries
        #self.collisionTranslations = [] #hold the translations for these moving boundaries
        self.collisionObjectCount = 0
        self.collisionObjectCenters = ti.Vector.field(2, dtype=float, shape=16) #allow up to 16 collision objects for now
        self.collisionVelocities = ti.Vector.field(2, dtype=float, shape=16) #hold the translations for these moving boundaries, we'll also use these to set vi for sticky bounds
        self.collisionTypes = ti.field(dtype=int, shape=16) #store collision types
        
        #Rankine Damage Parameters
        self.l0 = 0.5 * dx #as usual, close to the sqrt(2) * dx that they use
        self.Gf = 1.0
        self.Hs = ti.field(dtype=float, shape=self.numParticles) #weibull will make these different from eahc other
        self.useRankineDamageList = ti.field(dtype=int, shape=self.numParticles)

        #Time to Failure Damage Parameters, many of these will be set later when we add the damage model
        self.damageList = np.array(EList) #dummy list
        self.useTimeToFailureDamageList = ti.field(dtype=int, shape=self.numParticles)
        self.cf = 1.0
        self.timeToFail = 1.0
        self.sigmaF = ti.field(dtype=float, shape=self.numParticles) #each particle can have different sigmaF based on Weibull dist
        self.m = 1.0
        self.vRef = 1.0
        self.sigmaFRef = 1.0
        
        #Explicit MPM Fields
        self.x = ti.Vector.field(2, dtype=float, shape=self.numParticles) # position
        self.v = ti.Vector.field(2, dtype=float, shape=self.numParticles) # velocity
        self.C = ti.Matrix.field(2, 2, dtype=float, shape=self.numParticles) # affine velocity field
        self.F = ti.Matrix.field(2, 2, dtype=float, shape=self.numParticles) # deformation gradient
        self.material = ti.field(dtype=int, shape=self.numParticles) # material id
        self.mp = ti.field(dtype=float, shape=self.numParticles) # particle masses
        self.Vp = ti.field(dtype=float, shape=self.numParticles) # particle volumes
        self.Jp = ti.field(dtype=float, shape=self.numParticles) # plastic deformation
        self.grid_v = ti.Vector.field(2, dtype=float, shape=(self.nGrid, self.nGrid)) # grid node momentum/velocity
        self.grid_m = ti.field(dtype=float, shape=(self.nGrid, self.nGrid))  # grid node mass is nGrid x nGrid x 2, each grid node has a mass for each field
        self.gravity = ti.Vector.field(2, dtype=float, shape=())

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

    #------------Simulation Routines---------------

    #Simulation substep
    @ti.kernel
    def reinitializeGrid(self):
        #re-initialize grid quantities
        for i, j in self.grid_m:
            self.grid_v[i, j] = [0, 0]
            self.grid_m[i, j] = 0

    @ti.kernel
    def P2GandForces(self):
        # Particle state update and scatter to grid (P2G)
        for p in self.x: 
            
            #for particle p, compute base index
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            
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
            kirchoff = self.kirchoff_NeoHookean(self.F[p], J, self.mu[p], self.la[p])

            #P2G for velocity and mass AND Force Update!
            for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]
                
                dweight = ti.Vector.zero(float,2)
                dweight[0] = self.inv_dx * dw[i][0] * w[j][1]
                dweight[1] = self.inv_dx * w[i][0] * dw[j][1]
                
                force = -self.Vp[p] * kirchoff @ dweight

                #self.grid_v[base + offset] += self.mp[p] * weight * self.v[p] #momentum transfer (PIC)
                self.grid_v[base + offset] += self.mp[p] * weight * (self.v[p] + self.C[p] @ dpos) #momentum transfer
                self.grid_m[base + offset] += weight * self.mp[p] #mass transfer

                #TODO this is the line that explodes
                self.grid_v[base + offset] += self.dt * force #add force to update velocity, don't divide by mass bc this is actually updating MOMENTUM
    
    @ti.kernel
    def addGravity(self):
        # Gravity and Boundary Collision
        for i, j in self.grid_m:
            if self.grid_m[i, j] > 0: # No need for epsilon here
                self.grid_v[i, j] = (1 / self.grid_m[i, j]) * self.grid_v[i, j] # Momentum to velocity
                self.grid_v[i, j] += self.dt * self.gravity[None] # gravity

    @ti.kernel
    def G2P(self):
        # grid to particle (G2P)
        for p in self.x: 
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            new_F = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)): # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = self.grid_v[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]

                dweight = ti.Vector.zero(float,2)
                dweight[0] = self.inv_dx * dw[i][0] * w[j][1]
                dweight[1] = self.inv_dx * w[i][0] * dw[j][1]

                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
                new_F += g_v.outer_product(dweight)
                self.v[p], self.C[p] = new_v, new_C
                self.x[p] += self.dt * self.v[p] # advection
                self.F[p] = (ti.Matrix.identity(float, 2) + (self.dt * new_F)) @ self.F[p] #updateF (explicitMPM way)

    def substep(self):

        with Timer("Reinitialize Structures"):
            self.reinitializeGrid()
        with Timer("P2G and Forces"):
            self.P2GandForces()
        with Timer("Add Gravity"):
            self.addGravity()
        with Timer("Collision Objects"):
            for i in range(self.collisionObjectCount):
                t, v = self.transformCallbacks[i](self.elapsedTime) #get the current translation and velocity based on current time
                self.collisionVelocities[i] = ti.Vector(v)
                self.updateCollisionObjects(i)
                self.collisionCallbacks[i](i)
        with Timer("G2P"):
            self.G2P()

        self.elapsedTime += self.dt #update elapsed time

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
        self.collisionVelocities[self.collisionObjectCount] = ti.Vector([0.0, 0.0]) #add a dummy velocity for now
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
            for I in ti.grouped(self.grid_m):
                #treat as one field
                nodalMass = self.grid_m[I]
                if nodalMass > 0:
                    updatedCenter = self.collisionObjectCenters[id]
                    offset = I * self.dx - updatedCenter
                    n = ti.Vector(normal)
                    if offset.dot(n) < 0:
                        if self.collisionTypes[id] == self.surfaceSticky:
                            self.grid_v[I] = self.collisionVelocities[id] #set velocity to be the collision object's velocity
                        else:
                            v = self.grid_v[I] #divide out the mass to get velocity
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

                            self.grid_v[I] = v

        self.collisionCallbacks.append(collide)

    @ti.kernel
    def reset(self, arr: ti.ext_arr(), partCount: ti.ext_arr(), initVel: ti.ext_arr(), pMasses: ti.ext_arr(), pVols: ti.ext_arr(), EList: ti.ext_arr(), nuList: ti.ext_arr()):
        self.gravity[None] = [0, self.gravMag]
        for i in range(self.numParticles):
            self.x[i] = [ti.cast(arr[i,0], ti.f64), ti.cast(arr[i,1], ti.f64)]
            self.material[i] = 0
            self.v[i] = [0, 0]
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.Jp[i] = 1
            self.C[i] = ti.Matrix.zero(float, 2, 2)
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
                    self.v[idx] = [ti.cast(initVel[i,0], ti.f64), ti.cast(initVel[i,1], ti.f64)]
                    self.mp[idx] = ti.cast(pMasses[i], ti.f64)
                    self.Vp[idx] = ti.cast(pVols[i], ti.f64)
                    E = ti.cast(EList[i], ti.f64)
                    nu = ti.cast(nuList[i], ti.f64)
                    self.mu[idx] = E / (2 * (1 + nu))
                    self.la[idx] = E * nu / ((1+nu) * (1 - 2 * nu))
                    idx += 1 

        # #Now set up damage settings
        # #Compute Weibull Distributed SigmaF for TimeToFailure Model
        # for p in range(self.numParticles):
        #     if self.useTimeToFailureDamageList[p] == 1 and self.useRankineDamageList[p] == 1:
        #         ValueError('ERROR: you can only use one damage model at a time!')

        #     if(self.useTimeToFailureDamageList[p]):
        #         R = ti.cast(ti.random(ti.f32), ti.f64) #ti.random is broken for f64, so use f32
        #         self.sigmaF[p] = self.sigmaFRef * ( ((self.vRef * ti.log(R)) / (self.Vp[p] * ti.log(0.5)))**(1.0 / self.m) )

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
        if(s == -1):
            writer.export_frame(frame, self.outputPath)
        else:
            writer.export_frame(frame * self.numSubsteps + s, self.outputPath)

        #Construct positions for grid nodes
        gridX = np.zeros((self.nGrid**2, 2), dtype=float) #format as 1d array of nodal positions
        gridMasses = np.zeros((self.nGrid**2), dtype=float)
        gridVelocities = np.zeros((self.nGrid**2, 2), dtype=float)
        for i in range(self.nGrid):
            for j in range(self.nGrid):
                gridIdx = i * self.nGrid + j
                gridX[gridIdx,0] = i * self.dx
                gridX[gridIdx,1] = j * self.dx
                gridVelocities[gridIdx, 0] = self.grid_v[i, j][0]
                gridVelocities[gridIdx, 1] = self.grid_v[i, j][1]
                gridMasses[gridIdx] = self.grid_m[i,j]
        writer2 = ti.PLYWriter(num_vertices=self.nGrid**2)
        writer2.add_vertex_pos(gridX[:,0], gridX[:, 1], np.zeros(self.nGrid**2)) #add position
        writer2.add_vertex_channel("v_field1_x", "double", gridVelocities[:,0])
        writer2.add_vertex_channel("v_field1_y", "double", gridVelocities[:,1])
        writer2.add_vertex_channel("m", "double", gridMasses[:])

        if(s == -1):
            writer2.export_frame(frame, self.outputPath2)
        else:
            writer2.export_frame(frame * self.numSubsteps + s, self.outputPath2)

    def simulate(self):
        print("[Simulation] Particle Count: ", self.numParticles)
        print("[Simulation] Grid Dx: ", self.dx)
        print("[Simulation] Time Step: ", self.dt)
        self.reset(self.vertices, self.particleCounts, self.initialVelocity, self.pMasses, self.pVolumes, self.EList, self.nuList) #init
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