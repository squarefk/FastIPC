import taichi as ti
import numpy as np
import sys
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *

#ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu)  #CPU, parallel
ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

gravity = -10.0
outputPath = "../output/wabisabi/brittle.ply"
outputPath2 = "../output/wabisabi/brittle_nodes.ply"
fps = 30
endFrame = 3 * fps

E = 1e4 #1e5
nu = 0.15
EList = [E]
nuList = [nu]

rho = 1
filename = "../Data/TetMesh/bowl2_5k.mesh"
vertices, vol = sampleFromTetWild(filename, rho)
surfaceThreshold = 4.4

vertexCount = len(vertices)
particleCounts = [vertexCount]

pVol = vol / vertexCount
mp = pVol * rho
particleMasses = [mp]
particleVolumes = [pVol]

vel = 0.0
xVel = 0.0 
yVel = 0.0
zVel = 0.0
initVel = [xVel,yVel,zVel]
initialVelocity = [initVel]

#dx = 0.01 #TODO
ppd = 3 #particles per dimension
ppc = ppd**3 #want ppd per dimension
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

wallFriction = 0.1

groundCenter = (0, 0.05, 0)
groundNormal = (0, 1, 0)
surface = solver.surfaceSeparate
solver.addHalfSpace(groundCenter, groundNormal, surface, wallFriction)

groundCenter = (0, 0.95, 0)
groundNormal = (0, -1, 0)
surface = solver.surfaceSeparate
solver.addHalfSpace(groundCenter, groundNormal, surface, wallFriction)

groundCenter = (0.05, 0, 0)
groundNormal = (1, 0, 0)
surface = solver.surfaceSeparate
solver.addHalfSpace(groundCenter, groundNormal, surface, wallFriction)

groundCenter = (0.95, 0, 0)
groundNormal = (-1, 0, 0)
surface = solver.surfaceSeparate
solver.addHalfSpace(groundCenter, groundNormal, surface, wallFriction)

solver.simulate()