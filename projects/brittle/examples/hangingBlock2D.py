import taichi as ti
import numpy as np
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu)  #CPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

gravity = -10.0
outputPath = "../output/hangingBlock2D/"
fps = 24
endFrame = 10 * fps
vol = 0.2 * 0.2
ppc = 4
rho = 10.0
E, nu = 1000.0, 0.2 # Young's modulus and Poisson's ratio
EList = [E]
nuList = [nu]

#NOTE: surfaceThreshold tuning: 
#NOTE: 50 subdivs-- for 1 too low, 6 too high, 5 is perfect
#NOTE: 10 subdivs-- 5 is solid! gets the inner corner particles too, but seems fine
#NOTE: 70 subdivs-- 5 is also perfect omg!
#NOTE: I wonder if this seems to stay working because of our way of setting dx by ppc and what not
surfaceThreshold = 12.5

#Sample analtic box and get dx based on this distribution
minP = [0.4, 0.4]
maxP = [0.6, 0.6]
#vertices = sampleBoxGrid2D(minP, maxP, subdivs, 0, 0.5, 0.3)
vertices = sampleBox2D(minP, maxP)
particleCounts = [len(vertices)]
initialVelocity = [[0,0]]

pVol = vol / len(vertices)
particleMasses = [pVol * rho]
particleVolumes = [pVol]
pVol = vol / len(vertices)
dx = 0.01

#compute maxDt
cfl = 0.4
maxDt = suggestedDt(E, nu, rho, dx, cfl)
dt = 0.9 * maxDt

useDFG = True
verbose = False
useAPIC = False
symplectic = False
frictionCoefficient = 0.4
flipPicRatio = 0.90

if not symplectic: dt = 1e-3

prescoredDamageList = []
for i in range(len(vertices)):
    damage = 0.0
    if vertices[i][0] > 0.45 and vertices[i][1] > 0.546 and vertices[i][1] < 0.554:
        damage = 1.0
    if vertices[i][0] < 0.55 and vertices[i][1] > 0.446 and vertices[i][1] < 0.454:
        damage = 1.0
    prescoredDamageList.append(damage)

solver = DFGMPMSolver(endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, surfaceThreshold, useDFG, frictionCoefficient, verbose, useAPIC, flipPicRatio, symplectic, prescoredDamageList)

#Collision Objects
# groundCenter = (0, 0.05)
# groundNormal = (0, 1)
# groundCollisionType = solver.surfaceSlip
# solver.addHalfSpace(groundCenter, groundNormal, groundCollisionType, 0.0)

# leftWallCenter = (0.05, 0)
# leftWallNormal = (1, 0)
# leftWallCollisionType = solver.surfaceSlip
# solver.addHalfSpace(leftWallCenter, leftWallNormal, leftWallCollisionType, 0.0)

# rightWallCenter = (0.95, 0)
# rightWallNormal = (-1, 0)
# rightWallCollisionType = solver.surfaceSlip
# solver.addHalfSpace(rightWallCenter, rightWallNormal, rightWallCollisionType, 0.0)

ceilingCenter = (0, 0.59)
ceilingNormal = (0, -1)
ceilingCollisionType = solver.surfaceSticky
solver.addHalfSpace(ceilingCenter, ceilingNormal, ceilingCollisionType, 0.0)

solver.simulate()