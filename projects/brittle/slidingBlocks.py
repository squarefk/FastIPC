from sim.DFGMPMSolver import *
import taichi as ti
import numpy as np
from particleSampling import *
from cfl import suggestedDt
import math

#ti.init(arch=ti.gpu) # Try to run on GPU
ti.init(arch=ti.cpu, cpu_max_num_threads=1)

gravity = 0.0
outputPath = "output/slidingBlocks2D/brittle.ply"
outputPath2 = "output/slidingBlocks2D/brittle_nodes.ply"
fps = 24
endFrame = fps * 10
ppc = 9
rho = 10
E, nu = 1000.0, 0.2 # Young's modulus and Poisson's ratio

subdivs = 50
surfaceThreshold = 15 #5 and 5.5 seem good

#Sample analtic box and get dx based on this distribution
minP = [0.4, 0.4]
maxP = [0.6, 0.6]
vertices = sampleBoxGrid2D(minP, maxP, subdivs, 0, 0.5, 0.5)
box2 = sampleBoxGrid2D(minP, maxP, subdivs, 0, 0.73, 0.8)
particleCounts = [len(vertices), len(box2)]
vertices = np.concatenate((vertices, box2))

vol = 0.2 * 0.2 * math.pi
pVol = vol / len(vertices)
dx = (ppc * pVol)**0.5

#compute maxDt
cfl = 0.4
maxDt = suggestedDt(E, nu, rho, dx, cfl)
dt = 0.7 * maxDt

useFrictionalContact = True
verbose = False
useAPIC = False

initVel = [0.0,0.0]
initVel2 = [0.0, -1.0]
initialVelocity = [initVel, initVel2]

solver = DFGMPMSolver(endFrame, fps, dt, dx, E, nu, gravity, cfl, ppc, vol, rho, vertices, particleCounts, initialVelocity, outputPath, outputPath2, surfaceThreshold, useFrictionalContact, verbose, useAPIC)
solver.simulate()