import taichi as ti
import numpy as np
import sys
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

gravity = -10.0
outputPath = "../output/circleExplode2D/brittle.ply"
outputPath2 = "../output/circleExplode2D/brittle_nodes.ply"
fps = 240
endFrame = 3 * fps

E = 1e4 #1e5
nu = 0.15
EList = [E]
nuList = [nu]

N1 = 200
r1 = 0.07
centerPoint = [0.5, 0.5]


maxArea = 'qa0.0000025'
vertices = sampleCircle2D(centerPoint, r1, N1, maxArea)
surfaceThreshold = 4.4

vertexCount = len(vertices)
particleCounts = [vertexCount]

rho = 1 #6.9e-3
vol = r1 * r1 * math.pi
pVol = vol / vertexCount
mp = pVol * rho
particleMasses = [mp]
particleVolumes = [pVol]

vel = 0.0
xVel = vel 
yVel = -vel
initVel = [xVel,yVel]
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

if(len(sys.argv) == 6):
    outputPath = sys.argv[4]
    outputPath2 = sys.argv[5]

solver = DFGMPMSolver(endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThreshold, useDFG, frictionCoefficient, verbose, useAPIC, flipPicRatio)

#Add Damage Model
percentStretch = 7.5e-4 #7e-4 < p < 1e-3
dMin = 0.4
Gf = 5e-4 # 2.76e-7 before

#AnisoMPM Params
sigmaC = 30
p = 5e-2 #6e-3 < ? < 8e-3
eta = 1e-5
zeta = 1e4

if(len(sys.argv) == 6):
    percentStretch = float(sys.argv[1])
    Gf = float(sys.argv[2])
    dMin = float(sys.argv[3])

damageList = [1]
if useDFG == True: 
    #solver.addRankineDamage(damageList, percentStretch, Gf, dMin)
    solver.addAnisoMPMDamage(damageList, eta, dMin, percentStretch = p, zeta = zeta)

#Add Impulse
c = (0.5, 0.5)
strength = -1e4
startTime = 0.0
duration = 3.0 / float(fps)
if useDFG == True:
    solver.addImpulse(c, strength, startTime, duration)

useWeibull = False
vRef = vol
m = 10 #m6 with p 2.1e-5 gives red scatter throughout ring but no breaks, m3 has less damage happening, m8 has a break but weird behavior, m15 too red
if useWeibull == True: solver.addWeibullDistribution(vRef, m)

wallFriction = 0.1

groundCenter = (0, 0.05)
groundNormal = (0, 1)
surface = solver.surfaceSeparate
solver.addHalfSpace(groundCenter, groundNormal, surface, wallFriction)

groundCenter = (0, 0.95)
groundNormal = (0, -1)
surface = solver.surfaceSlip
solver.addHalfSpace(groundCenter, groundNormal, surface, wallFriction)

groundCenter = (0.05, 0)
groundNormal = (1, 0)
surface = solver.surfaceSlip
solver.addHalfSpace(groundCenter, groundNormal, surface, wallFriction)

groundCenter = (0.95, 0)
groundNormal = (-1, 0)
surface = solver.surfaceSlip
solver.addHalfSpace(groundCenter, groundNormal, surface, wallFriction)

solver.simulate()