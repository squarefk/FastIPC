import taichi as ti
import numpy as np
import sys
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

gravity = 0.0
outputPath = "../output/mode1Fracture/brittle.ply"
outputPath2 = "../output/mode1Fracture/brittle_nodes.ply"
fps = 30
endFrame = 2 * fps

E, nu = 1e4, 0.15 #TODO
EList = [E]
nuList = [nu]

st = 4.9  #10 for dx0.005, 
surfaceThreshold = st

maxArea = 'qa0.0000025'

grippedMaterial = 0.005
minPoint = [0.4, 0.4 - grippedMaterial]
maxPoint = [0.6, 0.6 + grippedMaterial]
vertices = sampleNotchedBox2D(minPoint, maxPoint, maxArea)
vertexCount = len(vertices)
particleCounts = [vertexCount]

rho = 1 #TODO
vol = 0.2 * (0.2 + (grippedMaterial*2))
pVol = vol / vertexCount
mp = pVol * rho
particleMasses = [mp]
particleVolumes = [pVol]

initVel = [0,0]
initialVelocity = [initVel]

ppc = 4   #TODO: 16 from ziran
dx = (ppc * pVol)**0.5
#dx = 0.005 #TODO: 0.005 from ziran

#Compute max dt
cfl = 0.4
maxDt = suggestedDt(E, nu, rho, dx, cfl)
dt = 0.9 * maxDt

useDFG = True
verbose = False
useAPIC = False
frictionCoefficient = 0.0
flipPicRatio = 0.9

# python3 script.py percentStretch eta zeta dMin outputPath1 outputPath2
if(len(sys.argv) == 6):
    outputPath = sys.argv[4]
    outputPath2 = sys.argv[5]

if(len(sys.argv) == 7):
    outputPath = sys.argv[5]
    outputPath2 = sys.argv[6]

solver = DFGMPMSolver(endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThreshold, useDFG, frictionCoefficient, verbose, useAPIC, flipPicRatio)

#Add Damage Model
# Gf = 0.01 #0.1 starts to get some red, but we wanna see it fast! TODO
# sigmaF = 89 # 87 < x < 89.5, 89 is solid!
# dMin = 0.25 #TODO, this controls how much damage must accumulate before we allow a node to separate

#Add Damage Model
#E=1e5, Gf=1e-3, p=1e-3 looks wild, lots of little fractures
percentStretch = 8e-4 # 1.12e-4 < p < 1.123e-4 NOTE: trying 2e-4 to see if we can get it to fracture later when it has some give to separate, (it was too low)
sigmaF = 33 #Gf1e-2: x < sigmaF < y ; Gf1e-4: 33 < sigmaF < 33.5 ; Gf1e-5: 10.572 < sigmaF < 10.575
dMin = 0.25
Gf = 1e-4 #1e-5

#AnisoMPM Params
sigmaC = 30
p = 1.7e-2 #1.5e-3 < ? < 1.8e-2
eta = 1e-4
zeta = 1e4

if(len(sys.argv) == 6):
    percentStretch = float(sys.argv[1])
    Gf = float(sys.argv[2])
    dMin = float(sys.argv[3])

if(len(sys.argv) == 7):
    p = float(sys.argv[1])
    eta = float(sys.argv[2])
    zeta = float(sys.argv[3])
    dMin = float(sys.argv[4])

damageList = [1]
if useDFG == True: 
    #solver.addRankineDamage(damageList,Gf, dMin, percentStretch = percentStretch)
    #solver.addRankineDamage(damageList,Gf, dMin, sigmaFRef = sigmaF)
    #solver.addAnisoMPMDamage(damageList, eta, dMin, sigmaC = sigmaC, zeta = zeta)
    solver.addAnisoMPMDamage(damageList, eta, dMin, percentStretch = p, zeta = zeta)

#Collision Objects
lowerCenter = (0.0, minPoint[1] + grippedMaterial)
lowerNormal = (0, 1)
upperCenter = (0.0, maxPoint[1] - grippedMaterial)
upperNormal = (0, -1)

def lowerTransform(time: ti.f64):
    translation = [0.0,0.0]
    velocity = [0.0,0.0]
    startTime = 0.0
    endTime = 4.0
    speed = -0.005
    if time >= startTime:
        translation = [0.0, speed * (time-startTime)]
        velocity = [0.0, speed]
    if time > endTime:
        translation = [0.0, 0.0]
        velocity = [0.0, 0.0]
    return translation, velocity

def upperTransform(time: ti.f64):
    translation = [0.0,0.0]
    velocity = [0.0,0.0]
    startTime = 0.0
    endTime = 4.0
    speed = 0.005
    if time >= startTime:
        translation = [0.0, speed * (time-startTime)]
        velocity = [0.0, speed]
    if time > endTime:
        translation = [0.0, 0.0]
        velocity = [0.0, 0.0]
    return translation, velocity

solver.addHalfSpace(lowerCenter, lowerNormal, surface = solver.surfaceSticky, friction = 0.0, transform = lowerTransform)
solver.addHalfSpace(upperCenter, upperNormal, surface = solver.surfaceSticky, friction = 0.0, transform = upperTransform)

#Simulate!
solver.simulate()