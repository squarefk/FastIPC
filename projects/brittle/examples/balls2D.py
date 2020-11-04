import taichi as ti
import numpy as np
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *
from projects.brittle.HalfSpace import *
import math

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=20) #CPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

gravity = -10.0
outputPath = "../output/balls2D/brittle.ply"
outputPath2 = "../output/balls2D/brittle_nodes.ply"
fps = 24
endFrame = fps * 10
ppc = 9
rho = 10
E, nu = 1000.0, 0.2 # Young's modulus and Poisson's ratio

surfaceThreshold = 10 #10 works

c1 = [0.2, 0.2]
c2 = [0.5, 0.2]
c3 = [0.8, 0.2]
c4 = [0.3, 0.5]
c5 = [0.7, 0.5]
c6 = [0.2, 0.8]
c7 = [0.5, 0.8]
c8 = [0.8, 0.8]
radius = 0.1
nSubDivs = 64
maxArea = 0.0001

#NOTE: maxArea is HARD CODED for sampleCircle2D because passing it as float doesn't work due to string conversion!!

vertices = sampleCircle2D(c1, radius, nSubDivs, maxArea)
circle2 = sampleCircle2D(c2, radius, nSubDivs, maxArea)
circle3 = sampleCircle2D(c3, radius, nSubDivs, maxArea)
circle4 = sampleCircle2D(c4, radius, nSubDivs, maxArea)
circle5 = sampleCircle2D(c5, radius, nSubDivs, maxArea)
circle6 = sampleCircle2D(c6, radius, nSubDivs, maxArea)
circle7 = sampleCircle2D(c7, radius, nSubDivs, maxArea)
circle8 = sampleCircle2D(c8, radius, nSubDivs, maxArea)
particleCounts = [len(vertices), len(circle2), len(circle3), len(circle4), len(circle5), len(circle6), len(circle7), len(circle8)]
vertices = np.concatenate((vertices, circle2))
vertices = np.concatenate((vertices, circle3))
vertices = np.concatenate((vertices, circle4))
vertices = np.concatenate((vertices, circle5))
vertices = np.concatenate((vertices, circle6))
vertices = np.concatenate((vertices, circle7))
vertices = np.concatenate((vertices, circle8))

vol = radius * radius * math.pi
pVol = vol / len(circle2)
dx = (ppc * pVol)**0.5

#compute maxDt
cfl = 0.4
maxDt = suggestedDt(E, nu, rho, dx, cfl)
dt = 0.7 * maxDt

useFrictionalContact = True
verbose = False
useAPIC = False

initVel = [0,0]
initialVelocity = []
particleMasses = []
particleVolumes = []
for i in range(8):
    initialVelocity.append(initVel)
    particleMasses.append(pVol * rho)
    particleVolumes.append(pVol)

solver = DFGMPMSolver(endFrame, fps, dt, dx, E, nu, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThreshold, useFrictionalContact, verbose, useAPIC)

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

#start sim!
solver.simulate()