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
endFrame = fps * 10
ppc = 9
rho = 10
E, nu = 50000.0, 0.2 # Young's modulus and Poisson's ratio

surfaceThreshold1 = 27 #too high --> too few surface particles
surfaceThreshold2 = 17

rampMin = [0.05, 0.05]
rampMax = [0.95, 0.2]
xDiff = rampMax[0] - rampMin[0]
yDiff = rampMax[1] - rampMin[1]
subdivs = 40
#vertices = sampleRamp2D(rampMin, rampMax, subdivs)

rampP1 = [0.05, 0.05]
rampP2 = [0.95, 0.05]
rampP3 = [0.05, 0.2]
vertices = sampleTriangle2D(rampP1, rampP2, rampP3)

boxMin = [0.4, 0.4]
boxMax = [0.5, 0.5]
boxSubdivs = 26
thetaRad = np.arctan(yDiff / xDiff)
boxTheta = -1 * np.degrees(thetaRad)
boxParticles = sampleBoxGrid2D(boxMin, boxMax, boxSubdivs, boxTheta, 0.2, 0.3)

particleCounts = [len(vertices), len(boxParticles)]
vertices = np.concatenate((vertices, boxParticles))

vol = (rampMax[0] - rampMin[0]) * (rampMax[1] - rampMin[1]) * 0.5
vol2 = (boxMax[0] - boxMin[0]) * (boxMax[1] - boxMin[1])
pVol = vol / particleCounts[0]
pVol2 = vol2 / particleCounts[1]
largerVol = max(pVol, pVol2)
#dx = (ppc * largerVol)**0.5
dx = 0.015

#compute maxDt
cfl = 0.4
maxDt = suggestedDt(E, nu, rho, dx, cfl)
dt = 0.7 * maxDt

useFrictionalContact = True
frictionCoefficient = 0.1
verbose = False
useAPIC = False

initialVelocity = [[0.0,0.0], [0.0,0.0]]
particleMasses = [pVol * rho, pVol2 * rho]
particleVolumes = [pVol, pVol2]
surfaceThresholds = [surfaceThreshold1, surfaceThreshold2]

solver = DFGMPMSolver(endFrame, fps, dt, dx, E, nu, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThresholds, useFrictionalContact, frictionCoefficient, verbose, useAPIC)
#solver = DFGMPMSolverWithPredefinedFields(endFrame, fps, dt, dx, E, nu, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThresholds, useFrictionalContact, frictionCoefficient, verbose, useAPIC)

#Collision Objects
groundCenter = (0, 0.05)
groundNormal = (0, 1)
groundCollisionType = solver.surfaceSticky
solver.addHalfSpace(groundCenter, groundNormal, groundCollisionType)

# leftWallCenter = (0.05, 0)
# leftWallNormal = (1, 0)
# leftWallCollisionType = solver.surfaceSlip
# solver.addHalfSpace(leftWallCenter, leftWallNormal, leftWallCollisionType)

rightWallCenter = (0.95, 0)
rightWallNormal = (-1, 0)
rightWallCollisionType = solver.surfaceSlip
solver.addHalfSpace(rightWallCenter, rightWallNormal, rightWallCollisionType)

# ceilingCenter = (0, 0.95)
# ceilingNormal = (0, -1)
# ceilingCollisionType = solver.surfaceSlip
# solver.addHalfSpace(ceilingCenter, ceilingNormal, ceilingCollisionType)

#start sim!
solver.simulate()