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
fps = 60
endFrame = 5 * fps

E, nu = 87 * 10**9, 0.17 #TODO
EList = [E]
nuList = [nu]

st = 10.0  #10.5 is solid, but it fractures numerically at the notch TODO
surfaceThreshold = st

maxArea = 'qpa0.0000025'

centerPoint = [0.5, 0.2]
N1 = 30
N2 = 16
r1 = 0.07
r2 = 0.055

vertices = sampleRing2D(centerPoint, r1, r2, N1, N2, maxArea)
vertexCount = len(vertices)
particleCounts = [vertexCount]

rho = 4000 #TODO
vol = (r1 * r1 * math.pi) - (r2 * r2 * math.pi)
pVol = vol / vertexCount
mp = pVol * rho
particleMasses = [mp]
particleVolumes = [pVol]

initVel = [0,-1.0]
initialVelocity = [initVel]

#dx = 0.01 #TODO
ppc = 8
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

if(len(sys.argv) == 6):
    outputPath = sys.argv[4]
    outputPath2 = sys.argv[5]

solver = DFGMPMSolver(endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThreshold, useDFG, frictionCoefficient, verbose, useAPIC, flipPicRatio)

#Add Damage Model
Gf = 2.3 #0.1 starts to get some red, but we wanna see it fast! TODO
sigmaF = 1e6 #for gf=2.3 and using weibull: ? < sigmaF < 5e6 
dMin = 0.25 #TODO, this controls how much damage must accumulate before we allow a node to separate

if(len(sys.argv) == 6):
    Gf = float(sys.argv[1])
    sigmaF = float(sys.argv[2])
    dMin = float(sys.argv[3])

damageList = [1]
if useDFG == True: solver.addRankineDamage(damageList, Gf, sigmaF, E, dMin)

useWeibull = True
sigmaFRef = sigmaF
vRef = vol
m = 6
if useWeibull == True: solver.addWeibullDistribution(sigmaFRef, vRef, m)

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