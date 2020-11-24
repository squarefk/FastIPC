import taichi as ti
import numpy as np
import sys
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

#General Sim Params
gravity = 0.0
outputPath = "../output/bulletShoot2D/brittle.ply"
outputPath2 = "../output/bulletShoot2D/brittle_nodes.ply"
fps = 60
endFrame = 5 * fps

#Elasticity Params
E, nu = 10000, 0.2 #TODO
EBullet, nuBullet = 5000, 0.2 #TODO
EList = [E, EBullet]
nuList = [nu, nuBullet]

#Surface Thresholding
st = 10.0  #TODO
stBullet = 10.0
surfaceThresholds = [st, stBullet]

#Particle Sampling
maxArea = 'qa0.0000025'

wallHeight = 0.6
wallThickness = 0.025
minPoint = [0.5 - (wallThickness/2.0), 0.5 - (wallHeight/2.0)]
maxPoint = [0.5 + (wallThickness/2.0), 0.5 + (wallHeight/2.0)]
vertices = sampleBox2D(minPoint, maxPoint, maxArea)
vertexCountWall = len(vertices)

bulletRadius = 0.03
distToWall = 0.1
bulletCenter = [minPoint[0] - distToWall, 0.5]
NBullet = 30
bulletVerts = sampleCircle2D(bulletCenter, bulletRadius, NBullet, maxArea)
vertexCountBullet = len(bulletVerts)

vertices = np.concatenate((vertices, bulletVerts)) #add the two particle handles together
particleCounts = [vertexCountWall, vertexCountBullet]

#Density, Volume, and Mass
rhoWall = 8 #TODO
volWall = wallThickness * wallHeight
pVolWall = volWall / vertexCountWall
mpWall = pVolWall * rhoWall

rhoBullet = 8 #TODO
volBullet = bulletRadius**2.0 * math.pi
pVolBullet = volBullet / vertexCountBullet
mpBullet = pVolBullet * rhoBullet

particleMasses = [mpWall, mpBullet]
particleVolumes = [pVolWall, pVolBullet]

#Initial Velocity
initVelWall = [0,0]
initVelBullet = [1, 0]
initialVelocity = [initVelWall, initVelBullet]

#Particle distribution and grid resolution
ppc = 8
#dx = (ppc * pVol)**0.5
dx = 0.005 #TODO

#Compute max dt
cfl = 0.4
maxDt = min(suggestedDt(E, nu, rhoWall, dx, cfl), suggestedDt(EBullet, nuBullet, rhoBullet, dx, cfl)) #take min of each suggested DT since we have two materials
dt = 0.9 * maxDt

useDFG = True
verbose = False
useAPIC = False
frictionCoefficient = 0.0
flipPicRatio = 0.9 #want to blend in more PIC for stiffness -> lower

solver = DFGMPMSolver(endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThresholds, useDFG, frictionCoefficient, verbose, useAPIC, flipPicRatio)

#Add Damage Model
Gf = 0.01 #0.1 starts to get some red, but we wanna see it fast! TODO
sigmaF = 88 #500 too high, TODO
dMin = 0.25 #TODO, this controls how much damage must accumulate before we allow a node to separate

damageList = [1, 0]
if useDFG == True: solver.addRankineDamage(damageList, Gf, sigmaF, E, dMin)

useWeibull = False
sigmaFRef = sigmaF
vRef = volWall
m = 6
if useWeibull = True: solver.addWeibullDistribution(sigmaFRef, vRef, m)

heldWallHeight = 0.05

groundCenter = (0, minPoint[1] + heldWallHeight)
groundNormal = (0, 1)
surface = solver.surfaceSticky
solver.addHalfSpace(groundCenter, groundNormal, surface, 0.0)

groundCenter = (0, maxPoint[1] - heldWallHeight)
groundNormal = (0, -1)
surface = solver.surfaceSticky
solver.addHalfSpace(groundCenter, groundNormal, surface, 0.0)

groundCenter = (0.05, 0)
groundNormal = (1, 0)
surface = solver.surfaceSlip
solver.addHalfSpace(groundCenter, groundNormal, surface, 0.0)

groundCenter = (0.95, 0)
groundNormal = (-1, 0)
surface = solver.surfaceSlip
solver.addHalfSpace(groundCenter, groundNormal, surface, 0.0)

solver.simulate()