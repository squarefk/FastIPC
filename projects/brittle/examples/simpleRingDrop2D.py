import taichi as ti
import numpy as np
import sys
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

gravity = -10.0
outputPath = "../output/ringDrop2D/brittle.ply"
outputPath2 = "../output/ringDrop2D/brittle_nodes.ply"
fps = 120
endFrame = 3 * fps

E, nu = 5e5, 0.2 #TODO
EList = [E]
nuList = [nu]

st = 4.3  #4.5 too low
surfaceThreshold = st

maxArea = 'qpa0.0000025'

centerPoint = [0.5, 0.14]
N1 = 200
N2 = 175
r1 = 0.07
r2 = 0.055

vertices = sampleRing2D(centerPoint, r1, r2, N1, N2, maxArea)
vertexCount = len(vertices)
particleCounts = [vertexCount]

rho = 2 #TODO
vol = (r1 * r1 * math.pi) - (r2 * r2 * math.pi)
pVol = vol / vertexCount
mp = pVol * rho
particleMasses = [mp]
particleVolumes = [pVol]

initVel = [0,-1.0]
initialVelocity = [initVel]

#dx = 0.01 #TODO
ppc = 4
dx = (ppc * pVol)**0.5

#Compute max dt
cfl = 0.4
maxDt = suggestedDt(E, nu, rho, dx, cfl)
dt = 0.9 * maxDt

useDFG = True
verbose = False
useAPIC = False
frictionCoefficient = 0.0
flipPicRatio = 0.9 #want to blend in more PIC for stiffness -> lower

if(len(sys.argv) == 5):
    outputPath = sys.argv[4]
    outputPath2 = sys.argv[5]

solver = DFGMPMSolver(endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThreshold, useDFG, frictionCoefficient, verbose, useAPIC, flipPicRatio)

#Add Damage Model
percentStretch = 2.1e-5 # 1.7e-5 < p < 2e-5
dMin = 0.4
Gf = 3e-6 #1e-6 < Gf < 1e-5 

if(len(sys.argv) == 5):
    percentStretch = float(sys.argv[1])
    dMin = float(sys.argv[2])

damageList = [1]
if useDFG == True: solver.addSimpleRankineDamage(damageList, percentStretch, dMin, Gf)

useWeibull = True
vRef = vol
m = 8 #m6 with p 2.1e-5 gives red scatter throughout ring but no breaks, m3 has less damage happening, m8 has a break but weird behavior, m15 too red
if useWeibull == True: solver.addSimpleWeibullDistribution(vRef, m)

groundCenter = (0, 0.05)
groundNormal = (0, 1)
surface = solver.surfaceSlip
solver.addHalfSpace(groundCenter, groundNormal, surface, 0.0)

groundCenter = (0, 0.95)
groundNormal = (0, -1)
surface = solver.surfaceSlip
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