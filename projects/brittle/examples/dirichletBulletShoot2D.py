import taichi as ti
import numpy as np
import sys
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *
from projects.brittle.DFGMPMSolver_Old import *

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

#General Sim Params
gravity = -10.0

outputPath = "../output/dirichletBulletShoot2D/NEWbrittle.ply"
outputPath2 = "../output/dirichletBulletShoot2D/NEWbrittle_nodes.ply"

fps = 120
endFrame = 3 * fps

#Elasticity Params
E, nu = 1e4, 0.15 #0.2
EList = [E]
nuList = [nu]

#Surface Thresholding
st = 4.8  #TODO
stBullet = st
surfaceThreshold = st

#Particle Sampling
maxAreaWall = 'qpa0.0000015'

wallHeight = 0.3
wallThickness = 0.05 #0.015
minPoint = [0.5 - (wallThickness/2.0), 0.5 - (wallHeight/2.0)]
maxPoint = [0.5 + (wallThickness/2.0), 0.5 + (wallHeight/2.0)]
vertices = sampleNotchedWall2D(minPoint, maxPoint, maxAreaWall)
vertexCountWall = len(vertices)
particleCounts = [vertexCountWall]

#Density, Volume, and Mass
rhoWall = 1 #8 #TODO
volWall = wallThickness * wallHeight
pVolWall = volWall / vertexCountWall
mpWall = pVolWall * rhoWall

particleMasses = [mpWall]
particleVolumes = [pVolWall]

#Initial Velocity
initVelWall = [0,0]
initialVelocity = [initVelWall]

#Particle distribution and grid resolution
ppc = 4
dx = (ppc * pVolWall)**0.5

#Compute max dt
cfl = 0.4
maxDt = suggestedDt(E, nu, rhoWall, dx, cfl) #take min of each suggested DT since we have two materials
dt = 0.9 * maxDt

useDFG = True
verbose = False
useAPIC = False
frictionCoefficient = 0.0
flipPicRatio = 0.9 #want to blend in more PIC for stiffness -> lower

if(len(sys.argv) == 6):
    outputPath = sys.argv[4]
    outputPath2 = sys.argv[5]

solver = DFGMPMSolver(endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThreshold, useDFG, frictionCoefficient, verbose, useAPIC, flipPicRatio)

#Add Damage Model
percentStretch = 1e-3 #9.5e-4
Gf = 5e-4 #5e-4
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

#Add Bullet Collision Object
def bulletTransform(time: ti.f64):
    translation = [0.0,0.0]
    velocity = [0.0,0.0]
    startTime = 0.0
    endTime = 2.0
    speed = 3.0
    if time >= startTime:
        translation = [speed * (time-startTime), 0.0]
        velocity = [speed, 0.0]
    if time > endTime:
        translation = [0.0, 0.0]
        velocity = [0.0, 0.0]
    return translation, velocity
bulletRadius = wallThickness * 0.9
distToWall = 0.07
bulletCenter = (minPoint[0] - distToWall, 0.5)
bulletCenter2 = (0.5, 0.5)
surface = solver.surfaceSeparate
surface2 = solver.surfaceSticky
solver.addSphereCollider(bulletCenter, bulletRadius, surface, transform = bulletTransform)
#solver.addSphereCollider(bulletCenter2, bulletRadius, surface2, transform = bulletTransform)

groundCenter = (0, minPoint[1] + heldWallHeight)
groundNormal = (0, 1)
surface = solver.surfaceSticky
#solver.addHalfSpace(groundCenter, groundNormal, surface, 0.0)
solver.addHalfSpace((0, minPoint[1] - heldWallHeight), groundNormal, solver.surfaceSeparate, friction = 0.1)

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