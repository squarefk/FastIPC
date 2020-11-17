import taichi as ti
import numpy as np
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel

gravity = -10.0
outputPath = "../output/hangingBlock2D/brittle.ply"
outputPath2 = "../output/hangingBlock2D/brittle_nodes.ply"
fps = 24
endFrame = 10 * fps
vol = 0.2 * 0.2
ppc = 4
rho = 1.0
E, nu = 1000.0, 0.2 # Young's modulus and Poisson's ratio

#NOTE: surfaceThreshold tuning: 
#NOTE: 50 subdivs-- for 1 too low, 6 too high, 5 is perfect
#NOTE: 10 subdivs-- 5 is solid! gets the inner corner particles too, but seems fine
#NOTE: 70 subdivs-- 5 is also perfect omg!
#NOTE: I wonder if this seems to stay working because of our way of setting dx by ppc and what not
subdivs = 50
surfaceThresholds = [5]

#Sample analtic box and get dx based on this distribution
minP = [0.4, 0.4]
maxP = [0.6, 0.6]
vertices = sampleBoxGrid2D(minP, maxP, subdivs, 0, 0.5, 0.3)
particleCounts = [len(vertices)]
initialVelocity = [[0,-1]]


pVol = vol / len(vertices)
particleMasses = [pVol * rho]
particleVolumes = [pVol]
pVol = vol / len(vertices)
dx = (ppc * pVol)**0.5

#compute maxDt
cfl = 0.4
maxDt = suggestedDt(E, nu, rho, dx, cfl)
dt = 0.7 * maxDt

useFrictionalContact = True
verbose = False
useAPIC = False
frictionCoefficient = 0.4
flipPicRatio = 0.95
useRankineDamage = False

solver = DFGMPMSolver(endFrame, fps, dt, dx, E, nu, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThresholds, useFrictionalContact, frictionCoefficient, verbose, useAPIC, flipPicRatio, useRankineDamage)

#Collision Objects
groundCenter = (0, 0.05)
groundNormal = (0, 1)
groundCollisionType = solver.surfaceSlip
solver.addHalfSpace(groundCenter, groundNormal, groundCollisionType)

leftWallCenter = (0.05, 0)
leftWallNormal = (1, 0)
leftWallCollisionType = solver.surfaceSlip
solver.addHalfSpace(leftWallCenter, leftWallNormal, leftWallCollisionType)

rightWallCenter = (0.95, 0)
rightWallNormal = (-1, 0)
rightWallCollisionType = solver.surfaceSlip
solver.addHalfSpace(rightWallCenter, rightWallNormal, rightWallCollisionType)

ceilingCenter = (0, 0.95)
ceilingNormal = (0, -1)
ceilingCollisionType = solver.surfaceSlip
solver.addHalfSpace(ceilingCenter, ceilingNormal, ceilingCollisionType)

solver.simulate()