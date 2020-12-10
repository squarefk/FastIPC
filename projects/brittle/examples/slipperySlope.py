import taichi as ti
import numpy as np
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *
from projects.brittle.DFGMPMSolverWithPredefinedFields import *
import math

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

gravity = -10
outputPath = "../output/slipperySlope2D/brittle.ply"
outputPath2 = "../output/slipperySlope2D/brittle_nodes.ply"
fps = 24
endFrame = fps * 5
ppc = 9
rho = 10
E, nu = 50000.0, 0.2 # Young's modulus and Poisson's ratio
EList = [E,E]
nuList = [nu, nu]
surfaceThreshold = 12

args = 'qa0.000005'

rampP1 = [0.05, 0.05]
rampP2 = [0.95, 0.05]
rampP3 = [0.05, 0.2]
yDiff = rampP3[1] - rampP1[1]
xDiff = rampP2[0] - rampP1[0]
vertices = sampleTriangle2D(rampP1, rampP2, rampP3, args)

boxMin = [0.4, 0.4]
boxMax = [0.5, 0.5]
boxSubdivs = 26
thetaRad = np.arctan(yDiff / xDiff)
boxTheta = -1 * np.degrees(thetaRad)
boxParticles = sampleTranslatedBox2D(boxMin, boxMax, boxSubdivs, boxTheta, 0.2, 0.27, args)

particleCounts = [len(vertices), len(boxParticles)]
vertices = np.concatenate((vertices, boxParticles))

vol = (xDiff) * (yDiff) * 0.5
vol2 = (boxMax[0] - boxMin[0]) * (boxMax[1] - boxMin[1])
pVol = vol / particleCounts[0]
pVol2 = vol2 / particleCounts[1]
largerVol = max(pVol, pVol2)
dx = (ppc * largerVol)**0.5

#compute maxDt
cfl = 0.4
maxDt = suggestedDt(E, nu, rho, dx, cfl)
dt = 0.7 * maxDt

useDFG = True
frictionCoefficient = 0.1
verbose = False
useAPIC = True
flipPicRatio = 0.0

initialVelocity = [[0.0,0.0], [0.0,0.0]]
particleMasses = [pVol * rho, pVol2 * rho]
particleVolumes = [pVol, pVol2]

solver = DFGMPMSolver(endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThreshold, useDFG, frictionCoefficient, verbose, useAPIC, flipPicRatio)

#Collision Objects
groundCenter = (0, 0.05)
groundNormal = (0, 1)
groundCollisionType = solver.surfaceSticky
solver.addHalfSpace(groundCenter, groundNormal, groundCollisionType, 0.0)

# leftWallCenter = (0.05, 0)
# leftWallNormal = (1, 0)
# leftWallCollisionType = solver.surfaceSlip
# solver.addHalfSpace(leftWallCenter, leftWallNormal, leftWallCollisionType)

rightWallCenter = (0.95, 0)
rightWallNormal = (-1, 0)
rightWallCollisionType = solver.surfaceSlip
solver.addHalfSpace(rightWallCenter, rightWallNormal, rightWallCollisionType, 0.0)

# ceilingCenter = (0, 0.95)
# ceilingNormal = (0, -1)
# ceilingCollisionType = solver.surfaceSlip
# solver.addHalfSpace(ceilingCenter, ceilingNormal, ceilingCollisionType)

#start sim!
solver.simulate()