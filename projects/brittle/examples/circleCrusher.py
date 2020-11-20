import taichi as ti
import numpy as np
import sys
from common.utils.particleSampling import *
from common.utils.cfl import *
from projects.brittle.DFGMPMSolver import *

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

gravity = 0.0
outputPath = "../output/circleCrusher/brittle.ply"
outputPath2 = "../output/circleCrusher/brittle_nodes.ply"
fps = 24
endFrame = 10 * fps

scale = 10**6

def computeEandNu(K, G):
    E = (9 * K * G) / (3*K + G)
    nu = (3*K - 2*G) / (2 * (3*K + G))
    return E, nu

K_d = 158.333 * 10**9 / scale #Pascals bulk mod TODO
G_d = 73.077 * 10**9 / scale#Pascals shear mod  TODO
E_d, nu_d = computeEandNu(K_d, G_d)

K_p = 260 * 10**9 / scale                       #TODO
G_p = 180 * 10**9 / scale                       #TODO
E_p, nu_p = computeEandNu(K_p, G_p)

EList = [E_d, E_p, E_p]
nuList = [nu_d, nu_p, nu_p]

st = 5
surfaceThresholds = [st, st, st]
maxArea = 'qa0.0000025'

c1 = [0.5, 0.5]
radius = 0.1
nSubDivs = 64
vertices = sampleCircle2D(c1, radius, nSubDivs, args = maxArea)
circleCount = len(vertices)

wp = 0.24
hp = 0.04
buffer = 0.01
platen1Min = [0.5 - (wp / 2.0), 0.5 + radius + buffer]
platen1Max = [0.5 + (wp/2.0), 0.5 + radius + hp + buffer]
box1 = sampleBox2D(platen1Min, platen1Max, args = maxArea)
box1Count = len(box1)
vertices = np.concatenate((vertices, box1))

platen2Min = [0.5 - (wp/2.0), 0.5 - radius - hp - buffer]
platen2Max = [0.5 + (wp/2.0), 0.5 - radius - buffer]
box2 = sampleBox2D(platen2Min, platen2Max, args = maxArea)
box2Count = len(box2)
vertices = np.concatenate((vertices, box2))

rho_d = 2300 #kg/m^-3 TODO
vol_d = radius * radius * math.pi
pVol_d = vol_d / circleCount
mp_d = pVol_d * rho_d

rho_p = 800 #kg/m^-3 TODO
vol_p = wp * hp
pVol_p1 = vol_p / box1Count
pVol_p2 = vol_p / box2Count
mp_p1 = pVol_p1 * rho_p
mp_p2 = pVol_p2 * rho_p

particleMasses = [mp_d, mp_p1, mp_p2]
particleVolumes = [pVol_d, pVol_p1, pVol_p2]
particleCounts = [circleCount, box1Count, box2Count]
initialVelocity = [[0,0],[0,0],[0,0]]
dx = 0.00362 #match the proportion in their demo  TODO
ppc = 9                                          #TODO

#compute maxDt
cfl = 0.4
maxDt = min(suggestedDt(E_d, nu_d, rho_d, dx, cfl), suggestedDt(E_p, nu_p, rho_p, dx, cfl)) #take the min between all objects suggestedDts
dt = 0.9 * maxDt

useFrictionalContact = True
verbose = False
useAPIC = False
frictionCoefficient = 0.4
flipPicRatio = 0.95

solver = DFGMPMSolver(endFrame, fps, dt, dx, EList, nuList, gravity, cfl, ppc, vertices, particleCounts, particleMasses, particleVolumes, initialVelocity, outputPath, outputPath2, surfaceThresholds, useFrictionalContact, frictionCoefficient, verbose, useAPIC, flipPicRatio)

print(len(sys.argv))
print('sigmaF: ', sys.argv[1])
print('cf: ', sys.argv[2])

#Add Damage Model
cf = 2000 / 10**5 / 2 #m/s                  TODO
sigmaFRef = 140 * 10**6 / scale #Pa TODO
vRef = 8 * 10**-6 / scale # m^3             TODO
m = 6.0
damageList = [1,0,0] #denote which objects should get damage
solver.addTimeToFailureDamage(damageList, cf, sigmaFRef, vRef, m)

#Collision Objects
grippedMaterial = 0.005
friction = 0.0
groundCenter = (0, 0.5 - radius - hp + grippedMaterial)
groundNormal = (0, 1)
groundCollisionType = solver.surfaceSticky #surfaceSlip

def lowerTransform(time: ti.f64):
    translation = [0.0,0.0]
    velocity = [0.0,0.0]
    startTime = 0.0
    endTime = 2.0
    speed = 0.005
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
    endTime = 2.0
    speed = -0.005
    if time >= startTime:
        translation = [0.0, speed * (time-startTime)]
        velocity = [0.0, speed]
    if time > endTime:
        translation = [0.0, 0.0]
        velocity = [0.0, 0.0]
    return translation, velocity

solver.addHalfSpace(groundCenter, groundNormal, groundCollisionType, 0.0, transform = lowerTransform)

leftWallCenter = (0.5 - (wp/2.0) - buffer, 0)
leftWallNormal = (1, 0)
leftWallCollisionType = solver.surfaceSlip
solver.addHalfSpace(leftWallCenter, leftWallNormal, leftWallCollisionType, 0.0)

rightWallCenter = (0.5 + (wp/2.0) + buffer, 0)
rightWallNormal = (-1, 0)
rightWallCollisionType = solver.surfaceSlip
solver.addHalfSpace(rightWallCenter, rightWallNormal, rightWallCollisionType, 0.0)

ceilingCenter = (0, 0.5 + radius + hp - grippedMaterial)
ceilingNormal = (0, -1)
ceilingCollisionType = solver.surfaceSticky
solver.addHalfSpace(ceilingCenter, ceilingNormal, ceilingCollisionType, 0.0, transform = upperTransform)

#solver.testEigenDecomp()
solver.simulate()