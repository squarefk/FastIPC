import taichi as ti
import numpy as np
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *
import math

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel

gravity = -10.0
outputPath = "../output/balls2D/brittle.ply"
outputPath2 = "../output/balls2D/brittle_nodes.ply"
fps = 24
endFrame = fps * 10
rho = 10
E, nu = 5000.0, 0.2 # Young's modulus and Poisson's ratio

surfaceThreshold = 4.5 #12 works for rad = 0.08 and 0.03

c1 = [0.14, 0.14]
c2 = [0.32, 0.14]
c3 = [0.5, 0.14]
c4 = [0.68, 0.14]
c5 = [0.86, 0.14]

c6 = [0.23, 0.32]
c7 = [0.41, 0.32]
c8 = [0.59, 0.32]
c9 = [0.77, 0.32]

c10 = [0.86, 0.5]
c11 = [0.14, 0.5]
c12 = [0.32, 0.5]
c13 = [0.5, 0.5]
c14 = [0.68, 0.5]

c15 = [0.23, 0.68]
c16 = [0.41, 0.68]
c17 = [0.59, 0.68]
c18 = [0.77, 0.68]

c19 = [0.68, 0.86]
c20 = [0.86, 0.86]
c21 = [0.14, 0.86]
c22 = [0.32, 0.86]
c23 = [0.5, 0.86]

# c24 = [0.68, 0.86]
# c25 = [0.86, 0.86]

radius = 0.03
nSubDivs = 64
maxArea = 'qa0.000005'

#NOTE: maxArea is HARD CODED for sampleCircle2D because passing it as float doesn't work due to string conversion!!

vertices = sampleCircle2D(c1, radius, nSubDivs, maxArea)
circle2 = sampleCircle2D(c2, radius, nSubDivs, maxArea)
circle3 = sampleCircle2D(c3, radius, nSubDivs, maxArea)
circle4 = sampleCircle2D(c4, radius, nSubDivs, maxArea)
circle5 = sampleCircle2D(c5, radius, nSubDivs, maxArea)
circle6 = sampleCircle2D(c6, radius, nSubDivs, maxArea)
circle7 = sampleCircle2D(c7, radius, nSubDivs, maxArea)
circle8 = sampleCircle2D(c8, radius, nSubDivs, maxArea)
circle9 = sampleCircle2D(c9, radius, nSubDivs, maxArea)
circle10 = sampleCircle2D(c10, radius, nSubDivs, maxArea)
circle11 = sampleCircle2D(c11, radius, nSubDivs, maxArea)
circle12 = sampleCircle2D(c12, radius, nSubDivs, maxArea)
circle13 = sampleCircle2D(c13, radius, nSubDivs, maxArea)
circle14 = sampleCircle2D(c14, radius, nSubDivs, maxArea)
circle15 = sampleCircle2D(c15, radius, nSubDivs, maxArea)
circle16 = sampleCircle2D(c16, radius, nSubDivs, maxArea)
circle17 = sampleCircle2D(c17, radius, nSubDivs, maxArea)
circle18 = sampleCircle2D(c18, radius, nSubDivs, maxArea)
circle19 = sampleCircle2D(c19, radius, nSubDivs, maxArea)
circle20 = sampleCircle2D(c20, radius, nSubDivs, maxArea)
circle21 = sampleCircle2D(c21, radius, nSubDivs, maxArea)
circle22 = sampleCircle2D(c22, radius, nSubDivs, maxArea)
circle23 = sampleCircle2D(c23, radius, nSubDivs, maxArea)
# circle24 = sampleCircle2D(c24, radius, nSubDivs, maxArea)
# circle25 = sampleCircle2D(c25, radius, nSubDivs, maxArea)

vertices = np.concatenate((vertices, circle2))
vertices = np.concatenate((vertices, circle3))
vertices = np.concatenate((vertices, circle4))
vertices = np.concatenate((vertices, circle5))
vertices = np.concatenate((vertices, circle6))
vertices = np.concatenate((vertices, circle7))
vertices = np.concatenate((vertices, circle8))
vertices = np.concatenate((vertices, circle9))
vertices = np.concatenate((vertices, circle10))
vertices = np.concatenate((vertices, circle11))
vertices = np.concatenate((vertices, circle12))
vertices = np.concatenate((vertices, circle13))
vertices = np.concatenate((vertices, circle14))
vertices = np.concatenate((vertices, circle15))
vertices = np.concatenate((vertices, circle16))
vertices = np.concatenate((vertices, circle17))
vertices = np.concatenate((vertices, circle18))
vertices = np.concatenate((vertices, circle19))
vertices = np.concatenate((vertices, circle20))
vertices = np.concatenate((vertices, circle21))
vertices = np.concatenate((vertices, circle22))
vertices = np.concatenate((vertices, circle23))
# vertices = np.concatenate((vertices, circle24))
# vertices = np.concatenate((vertices, circle25))

vol = radius * radius * math.pi
pVol = vol / len(circle2)
ppc = 4
dx = (ppc * pVol)**0.5

#compute maxDt
cfl = 0.4
maxDt = suggestedDt(E, nu, rho, dx, cfl)
dt = 0.7 * maxDt

useDFG = True
frictionCoefficient = 0.4
verbose = False
useAPIC = False
flipPicRatio = 0.95 #0 for full PIC, 1 for full FLIP

initVel = [0,0]
initialVelocity = []
particleMasses = []
particleVolumes = []
surfaceThresholds = []
particleCounts = []
EList = []
nuList = []
for i in range(23):
    particleCounts.append(len(circle2))
    initialVelocity.append(initVel)
    particleMasses.append(pVol * rho)
    particleVolumes.append(pVol)
    surfaceThresholds.append(surfaceThreshold)
    EList.append(E)
    nuList.append(nu)

solver = DFGMPMSolver(endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThreshold, useDFG, frictionCoefficient, verbose, useAPIC, flipPicRatio)

#Collision Objects
groundCenter = (0, 0.05)
groundNormal = (0, 1)
groundCollisionType = solver.surfaceSlip
solver.addHalfSpace(groundCenter, groundNormal, groundCollisionType, 0.0)

leftWallCenter = (0.05, 0)
leftWallNormal = (1, 0)
leftWallCollisionType = solver.surfaceSlip
solver.addHalfSpace(leftWallCenter, leftWallNormal, leftWallCollisionType, 0.0)

rightWallCenter = (0.95, 0)
rightWallNormal = (-1, 0)
rightWallCollisionType = solver.surfaceSlip
solver.addHalfSpace(rightWallCenter, rightWallNormal, rightWallCollisionType, 0.0)

ceilingCenter = (0, 0.95)
ceilingNormal = (0, -1)
ceilingCollisionType = solver.surfaceSlip
solver.addHalfSpace(ceilingCenter, ceilingNormal, ceilingCollisionType, 0.0)

#start sim!
solver.simulate()