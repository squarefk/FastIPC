import taichi as ti
import numpy as np
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel

gravity = -10.0
outputPath = "../output/circleCrusher/brittle.ply"
outputPath2 = "../output/circleCrusher/brittle_nodes.ply"
fps = 24
endFrame = 10 * fps
ppc = 9

rho = 2300 #kg/m^-3
K = 158.333 * 10**9 #Pascals bulk mod
G = 73.077 * 10**9 #Pascals shear mod
#use K and G to compute E and nu:
E = (9 * K * G) / (3*K + G)
nu = (3*K - 2*G) / (2 * (3*K + G))

surfaceThresholds = [15]

c1 = [0.5, 0.3]
radius = 0.1
nSubDivs = 64
maxArea = 0.0001
vertices = sampleCircle2D(c1, radius, nSubDivs, maxArea)

vol = radius * radius * math.pi
pVol = vol / len(vertices)
particleMasses = [pVol * rho]
particleVolumes = [pVol]
dx = 0.01

#compute maxDt
cfl = 0.4
maxDt = suggestedDt(E, nu, rho, dx, cfl)
dt = 0.7 * maxDt

useFrictionalContact = True
verbose = False
useAPIC = False
frictionCoefficient = 0.4
flipPicRatio = 0.95
useRankineDamage = True
sigmaCrit = 140 * 10**6 #Pa
Gf = 1.0
cf = 2000 #m/s

particleCounts = [len(vertices)]
initialVelocity = [[0,0]]

solver = DFGMPMSolver(endFrame, fps, dt, dx, E, nu, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThresholds, useFrictionalContact, frictionCoefficient, verbose, useAPIC, flipPicRatio, useRankineDamage, cf)

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