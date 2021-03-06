import taichi as ti
import numpy as np
import sys
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *
from projects.brittle.DFGMPMSolver_Old import *

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

useOld = False

#General Sim Params
gravity = 0.0

outputPath = "../output/bulletShoot2D/"

fps = 120
endFrame = 1 * fps

#Elasticity Params
E, nu = 1e5, 0.17 #0.2 #TODO
EBullet, nuBullet = 5e3, 0.2 #TODO
EList = [E, EBullet]
nuList = [nu, nuBullet]

#Surface Thresholding
st = 4.8  #TODO
stBullet = st
surfaceThreshold = st

#Particle Sampling
maxAreaWall = 'qpa0.0000015'
maxAreaBullet = 'qa0.0000015'

wallHeight = 0.3
wallThickness = 0.05 #0.015
minPoint = [0.5 - (wallThickness/2.0), 0.5 - (wallHeight/2.0)]
maxPoint = [0.5 + (wallThickness/2.0), 0.5 + (wallHeight/2.0)]
vertices = sampleNotchedWall2D(minPoint, maxPoint, maxAreaWall)
vertexCountWall = len(vertices)

bulletRadius = wallThickness * 0.9
distToWall = 0.07
bulletCenter = [minPoint[0] - distToWall, 0.5]
NBullet = 50
bulletVerts = sampleCircle2D(bulletCenter, bulletRadius, NBullet, maxAreaBullet)
vertexCountBullet = len(bulletVerts)

vertices = np.concatenate((vertices, bulletVerts)) #add the two particle handles together
particleCounts = [vertexCountWall, vertexCountBullet]

#Density, Volume, and Mass
rhoWall = 6.9e-3 #8 #TODO
volWall = wallThickness * wallHeight
pVolWall = volWall / vertexCountWall
mpWall = pVolWall * rhoWall

rhoBullet = 8e-3 #TODO
volBullet = bulletRadius**2.0 * math.pi
pVolBullet = volBullet / vertexCountBullet
mpBullet = pVolBullet * rhoBullet

particleMasses = [mpWall, mpBullet]
particleVolumes = [pVolWall, pVolBullet]

#Initial Velocity
xVel = 1.0
initVelWall = [0,0]
initVelBullet = [xVel, 0]
initialVelocity = [initVelWall, initVelBullet]

#Particle distribution and grid resolution
ppc = 4
dx = (ppc * pVolWall)**0.5
#dx = 0.005 #TODO

#Compute max dt
cfl = 0.4
maxDt = min(suggestedDt(E, nu, rhoWall, dx, cfl), suggestedDt(EBullet, nuBullet, rhoBullet, dx, cfl)) #take min of each suggested DT since we have two materials
dt = 0.9 * maxDt

useDFG = True
verbose = False
useAPIC = False
frictionCoefficient = 0.0
flipPicRatio = 0.9 #want to blend in more PIC for stiffness -> lower

if(len(sys.argv) == 6):
    outputPath = sys.argv[4]
    outputPath2 = sys.argv[5]

solver = DFGMPMSolver(endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, surfaceThreshold, useDFG, frictionCoefficient, verbose, useAPIC, flipPicRatio)
if useOld:
    solver = DFGMPMSolverOLD(endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThreshold, useDFG, frictionCoefficient, verbose, useAPIC, flipPicRatio)

#Add Damage Model
percentStretch = 1e-4 #9e-5 < p < ?
Gf = 5e-6 #5e-7 < Gf < 6e-6   5.5E-6 doesnt break still
dMin = 0.4 #TODO, this controls how much damage must accumulate before we allow a node to separate

if(len(sys.argv) == 6):
    percentStretch = float(sys.argv[1])
    Gf = float(sys.argv[2])
    dMin = float(sys.argv[3])

damageList = [1, 0]
if useDFG == True: solver.addRankineDamage(damageList, percentStretch, Gf, dMin)

useWeibull = False
vRef = volWall
m = 6
if useWeibull == True: solver.addWeibullDistribution(vRef, m)


#Add Collision Objects
heldWallHeight = 0.05

groundCenter = (0, minPoint[1] + heldWallHeight)
groundNormal = (0, 1)
surface = solver.surfaceSticky
#solver.addHalfSpace(groundCenter, groundNormal, surface, 0.0)

groundCenter = (0, maxPoint[1] - heldWallHeight)
groundNormal = (0, -1)
surface = solver.surfaceSticky
#solver.addHalfSpace(groundCenter, groundNormal, surface, 0.0)

groundCenter = (0.05, 0)
groundNormal = (1, 0)
surface = solver.surfaceSlip
solver.addHalfSpace(groundCenter, groundNormal, surface, 0.0)

groundCenter = (0.95, 0)
groundNormal = (-1, 0)
surface = solver.surfaceSlip
solver.addHalfSpace(groundCenter, groundNormal, surface, 0.0)

#Simulate!
solver.simulate()