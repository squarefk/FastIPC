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
fps = 240
endFrame = 240

E = 1e4 #1e5
nu = 0.15
EList = [E]
nuList = [nu]

st = 4.3  #4.5 too low
surfaceThreshold = st

maxArea = 'qpa0.0000025'

N1 = 200
N2 = 175
r1 = 0.07
r2 = 0.055
#centerPoint = [0.5, 0.14]
# centerPoint = [0.95 - r1 - 0.01, 0.15]
centerPoint = [0.95 - 2.1*r1, 0.05 + 1.1*r1]
useDisk = False

vertices = sampleRing2D(centerPoint, r1, r2, N1, N2, maxArea)

if useDisk:
    maxArea = 'qa0.0000025'
    vertices = sampleCircle2D(centerPoint, r1, N1, maxArea)
    surfaceThreshold = 4.4

vertexCount = len(vertices)
particleCounts = [vertexCount]

rho = 1 #6.9e-3
vol = (r1 * r1 * math.pi) - (r2 * r2 * math.pi)
if useDisk:
    vol = r1 * r1 * math.pi
pVol = vol / vertexCount
mp = pVol * rho
particleMasses = [mp]
particleVolumes = [pVol]

vel = 3.0
xVel = vel 
yVel = -vel
if useDisk == False:
    xVel = 0.0
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
#w/o weibl minP: 2.4750232950e-3 RING
#w/o weibl maxP: 2.4750232955e-3 RING
#w weibull minP: 1e-3 gives a little bit of breakage
#w weibull maxP: 1.2e-3
#w/o weibl minP: 2.85e-3 DISK 2.8 looks like beginning to have some cracks??
#w/o weibl maxP: 3.0e-3 DISK
percentStretch = 7.5e-4 #7e-4 < p < 1e-3
dMin = 0.4
Gf = 5e-4 # 2.76e-7 before
# 2.64e-6 is proportional to E for porcelain ;1e-3 was what we were testing on 12/4/20

if(len(sys.argv) == 6):
    percentStretch = float(sys.argv[1])
    Gf = float(sys.argv[2])
    dMin = float(sys.argv[3])

damageList = [1]
if useDFG == True: solver.addRankineDamage(damageList, percentStretch, Gf, dMin)

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
if useDisk == False:
    groundCenter = (centerPoint[0] - 2.1*r1, 0)
groundNormal = (1, 0)
surface = solver.surfaceSlip
solver.addHalfSpace(groundCenter, groundNormal, surface, wallFriction)

groundCenter = (0.95, 0)
groundNormal = (-1, 0)
surface = solver.surfaceSlip
solver.addHalfSpace(groundCenter, groundNormal, surface, wallFriction)

solver.simulate()