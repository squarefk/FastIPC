import taichi as ti
import numpy as np
from particleSampling import *
from cfl import suggestedDt

ti.init(arch=ti.gpu) # Try to run on GPU
#ti.init(arch=ti.cpu, cpu_max_num_threads=1)

##########

#Particle Sampling

#2D Square (using Triangle)
minP = [0.4, 0.4]
maxP = [0.6, 0.6]
maxArea = 0.0001
#vertices = sampleBox2D(minP, maxP, maxArea)

#2D Square (using OBJ)
#vertices = readOBJ("Data/OBJs/square2D.obj")

#2D Circle (using Triangle)
centerPoint = [0.5, 0.5]
radius = 0.1
nSubDivs = 64
maxArea = 0.0001
vertices = sampleCircle2D(centerPoint, radius, nSubDivs, maxArea)
circle2 = sampleCircle2D([0.7, 0.5], radius/2.0, nSubDivs, maxArea)
circle3 = sampleCircle2D([0.3, 0.5], radius/2.0, nSubDivs, maxArea)
particleCount = [len(vertices), len(circle2), len(circle3)]
# vertices = np.concatenate((vertices, circle2))
# vertices = np.concatenate((vertices, circle3))

#analytic box
vertices = sampleBoxGrid2D(minP, maxP, 50)

##########

#Simulation Variables
dim = 2
n_particles = len(vertices)
p_vol = (0.2 * 0.2) / n_particles
ppc = 4 if dim == 2 else 8
dx = (ppc * p_vol)**0.5 if dim == 2 else (ppc * p_vol)**(1.0/3.0)
inv_dx = 1.0 / dx 
n_grid = ti.ceil(inv_dx)
p_rho = 1.0
p_mass = p_vol * p_rho
E, nu = 1e3, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters

cfl = 0.4
maxDt = suggestedDt(E, nu, p_rho, dx, cfl)
dt = 0.8 * maxDt

print("[Simulation] Particle Count: ", n_particles)
print("[Simulation] Grid Dx: ", dx)
print("[Simulation] Time Step: ", dt)

#Neighbor Search Variables
#NOTE: if these values are too low, we get data races!!! Try to keep these as high as possible (RIP to ur RAM)
maxNeighbors = 1024
maxPPC = 1024

#DFG Parameters
rp = (3*(dx**2))**0.5 if dim == 3 else (2*(dx**2))**0.5 #set rp based on dx (this changes if dx != dy)
maxParticlesInfluencingGridNode = 4 * maxPPC #2d = 4*ppc, 3d = 8*ppc
dMin = 0.25
fricCoeff = 0.0
epsilon_m = 0.0001

##########

#Explicit MPM Fields
x = ti.Vector.field(2, dtype=float, shape=n_particles) # position
v = ti.Vector.field(2, dtype=float, shape=n_particles) # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles) # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles) # deformation gradient
material = ti.field(dtype=int, shape=n_particles) # material id
Jp = ti.field(dtype=float, shape=n_particles) # plastic deformation
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid, 2)) # grid node momentum/velocity, store two vectors at each grid node (for each field)
grid_m = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid)) # grid node mass is nGrid x nGrid x 2, each grid node has a mass for each field
gravity = ti.Vector.field(2, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())

#DFG Fields
Dp = ti.field(dtype=float, shape=n_particles) #particle damage
particleDG = ti.Vector.field(2, dtype=float, shape=n_particles) #keep track of particle damage gradients
gridDG = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid)) #grid node damage gradients
maxHelperCount = ti.field(int)
maxHelper = ti.field(int)
ti.root.dense(ti.ij, (n_grid,n_grid)).place(maxHelperCount) #maxHelperCount is nGrid x nGrid and keeps track of how many candidates we have for the new nodeDG maximum
ti.root.dense(ti.ij, (n_grid,n_grid)).dense(ti.k, maxParticlesInfluencingGridNode).place(maxHelper) #maxHelper is nGrid x nGrid x maxParticlesInfluencingGridNode
gridSeparability = ti.Vector.field(4, dtype=float, shape=(n_grid, n_grid)) # grid separability is nGrid x nGrid x 4, each grid node has a seperability condition for each field and we need to add up the numerator and denominator
gridMaxDamage = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid)) # grid max damage is nGrid x nGrid x 2, each grid node has a max damage from each field
separable = ti.field(dtype=int, shape=(n_grid,n_grid)) # whether grid node is separable or not
grid_n = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid, 2)) # grid node normals for two field nodes, store two vectors at each grid node (one for each field)
grid_f = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid, 2)) # grid node forces for two field nodes, store two vectors at each grid node (one for each field)


#Active Fields
activeFields = ti.field(int)
activeFieldsCount = ti.field(int)
ti.root.dense(ti.ij, (n_grid,n_grid)).dense(ti.k, 2).dense(ti.l, maxParticlesInfluencingGridNode).place(activeFields) #activeFields is nGrid x nGrid x 2 x numParticlesMappingToGridNode
ti.root.dense(ti.ij, (n_grid,n_grid)).dense(ti.k, 2).place(activeFieldsCount) #activeFieldsCount is nGrid x nGrid x 2 to hold counters for the active field lists
particleAF = ti.Vector.field(9, dtype=float, shape=n_particles) #store which activefield each particle belongs to for the 9 grid nodes it maps to

#Neighbor Search Fields
gridNumParticles = ti.field(int)      #track number of particles in each cell using cell index
backGrid = ti.field(int)              #background grid to map grid cells to a list of particles they contain
particleNumNeighbors = ti.field(int)  #track how many neighbors each particle has
particleNeighbors = ti.field(int)     #map a particle to its list of neighbors
#Shape the neighbor fields
gridShape = ti.root.dense(ti.ij, (n_grid,n_grid))
gridShape.place(gridNumParticles) #gridNumParticles is nGrid x nGrid
gridShape.dense(ti.k, maxPPC).place(backGrid) #backGrid is nGrid x nGrid x maxPPC
#NOTE: backgrid uses nGrid x nGrid even though dx != rp, but rp is ALWAYs larger than dx so it's okay!
particleShape = ti.root.dense(ti.i, n_particles)
particleShape.place(particleNumNeighbors) #particleNumNeighbors is nParticles x 1
particleShape.dense(ti.j, maxNeighbors).place(particleNeighbors) #particleNeighbors is nParticles x maxNeighbors

##########

#Constitutive Model
@ti.func 
def kirchoff_FCR(F, R, J, mu, la):
  #compute Kirchoff stress using FCR elasticity
  return 2 * mu * (F - R) @ F.transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1) #compute kirchoff stress for FCR model (remember tau = P F^T)

##########

#Neighbor Search Routines
@ti.func
def backGridIdx(x):
  #compute int vector of backGrid indeces (recall the grid here )
  return int(x/rp)

@ti.func
def isInGrid(x):
  return 0 <= x[0] and x[0] < n_grid and 0 <= x[1] and x[1] < n_grid

##########

#DFG Function Evaluations
@ti.func
def computeOmega(rBar): 
  #compute kernel function
  return 1 - (3 * (rBar**2)) + (2 * (rBar**3)) if (rBar >= 0 and rBar <= 1) else 0

@ti.func
def computeOmegaPrime(rBar): 
  #compute kernel function derivative
  return 6*(rBar**2 - rBar) if (rBar >= 0 and rBar <= 1) else 0

@ti.func
def computeRBar(x, xp): 
  #compute normalized distance
  return (x-xp).norm() / rp

@ti.func
def computeRBarGrad(x, xp): 
  #compute gradient of normalized distance
  return (x - xp) * (1 / (rp * (x-xp).norm()))

##########

#Simulation Routines
@ti.kernel
def substep():
  
  #re-initialize grid quantities
  for i, j in grid_m:
    grid_v[i, j, 0] = [0, 0] #field 1 vel
    grid_v[i, j, 1] = [0, 0] #field 2 vel
    grid_n[i, j, 0] = [0, 0] #field 1 normal
    grid_n[i, j, 1] = [0, 0] #field 2 normal
    grid_f[i, j, 0] = [0, 0] #f1 nodal force
    grid_f[i, j, 1] = [0, 0] #f2 nodal force
    grid_m[i, j] = [0, 0] #stacked to contain mass for each field
    gridSeparability[i, j] = [0, 0, 0, 0] #stackt fields, and we use the space to add up the numerator and denom for each field
    gridMaxDamage[i, j] = [0, 0] #stackt fields
    gridDG[i, j] = [0, 0] #reset grid node damage gradients
    separable[i,j] = -1 #-1 for only one field, 0 for not separable, and 1 for separable
  
  #Clear neighbor look up structures as well as maxHelperCount and activeFieldsCount
  for I in ti.grouped(gridNumParticles):
    gridNumParticles[I] = 0
  for I in ti.grouped(particleNeighbors):
    particleNeighbors[I] = -1
  for I in ti.grouped(maxHelperCount):
    maxHelperCount[I] = 0
  for I in ti.grouped(activeFieldsCount):
    activeFieldsCount[I] = 0
  for I in ti.grouped(particleAF):
    particleAF[I] = [-1, -1, -1, -1, -1, -1, -1, -1, -1]

  #Sort particles into backGrid
  for serial in range(1): #serialize this to maintain neighbor order... (it has a parallelism bug and idk why)
    for p in range(n_particles):
      cell = backGridIdx(x[p]) #grab cell idx (vector of ints)
      offs = ti.atomic_add(gridNumParticles[cell], 1) #atomically add one to our grid cell's particle count NOTE: returns the OLD value before add
      backGrid[cell, offs] = p #place particle idx into the grid cell bucket at the correct place in the cell's neighbor list (using offs)

  #Sort into particle neighbor lists
  #See https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py for reference
  for p_i in range(n_particles):
    pos = x[p_i]
    cell = backGridIdx(pos)
    nb_i = 0
    for offs in ti.static(ti.grouped(ti.ndrange((-1,2),(-1,2)))):
      cellToCheck = cell + offs
      if isInGrid(cellToCheck):
        for j in range(gridNumParticles[cellToCheck]): #iterate over all particles in this cell
          p_j = backGrid[cellToCheck, j]
          if nb_i < maxNeighbors and p_j != p_i and (pos - x[p_j]).norm() < rp:
            particleNeighbors[p_i, nb_i] = p_j
            nb_i += 1
    particleNumNeighbors[p_i] = nb_i

  #Compute DG for all particles and for all grid nodes 
  # NOTE: grid node DG is based on max of mapped particle DGs, in this loop we simply create a list of candidates, then we will take max after
  for p in range(n_particles):
    pos = x[p]
    DandS = ti.Vector([0.0, 0.0]) #hold D and S in here (no temporary atomic scalar variables in taichi...)
    nablaD = ti.Vector([0.0, 0.0])
    nablaS = ti.Vector([0.0, 0.0])
    for i in range(particleNumNeighbors[p]): #iterate over neighbors of particle p

      xp_i = particleNeighbors[p, i] #index of curr neighbor
      xp = x[xp_i] #grab neighbor position
      rBar = computeRBar(pos, xp)
      rBarGrad = computeRBarGrad(pos, xp)
      omega = computeOmega(rBar)
      omegaPrime = computeOmegaPrime(rBar)

      #Add onto the sums for D, S, nablaD, nablaS
      deltaDS = ti.Vector([(Dp[xp_i] * omega), omega])
      DandS += deltaDS
      nablaD += (Dp[xp_i] * omegaPrime * rBarGrad)
      nablaS += (omegaPrime * rBarGrad)

    nablaDBar = (nablaD * DandS[1] - DandS[0] * nablaS) / (DandS[1]**2) #final particle DG
    particleDG[p] = nablaDBar #store particle DG

    #Now iterate over the grid nodes particle p maps to to set Di!
    base = (pos * inv_dx - 0.5).cast(int)
    for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j])
      gridIdx = base + offset
      currGridDG = gridDG[gridIdx] # grab current grid Di

      if(nablaDBar.norm() > currGridDG.norm()): #save this particle's index as a potential candidate for new maximum
        offs = ti.atomic_add(maxHelperCount[gridIdx], 1) #this lets us keep a dynamically sized list by tracking the index
        maxHelper[gridIdx, offs] = p #add this particle index to the list!

  #Now iterate over all active grid nodes and compute the maximum of candidate DGs
  for i, j in maxHelperCount:
    currMaxDG = gridDG[i,j] # grab current grid Di
    currMaxNorm = currMaxDG.norm()

    for k in range(maxHelperCount[i,j]):
      p_i = maxHelper[i,j,k] #grab particle index
      candidateDG = particleDG[p_i]
      if(candidateDG.norm() > currMaxNorm):
        currMaxNorm = candidateDG.norm()
        currMaxDG = candidateDG

    gridDG[i,j] = currMaxDG #set to be the max we found

  # P2G for mass, set active fields, and compute separability conditions
  for p in x: 
    
    #for particle p, compute base index
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    
    # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

    #P2G for mass, set active fields, and compute separability conditions
    for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j])
      gridIdx = base + offset
      dpos = (offset.cast(float) - fx) * dx
      weight = w[i][0] * w[j][1]

      #Set Active Fields for each grid node! 
      if particleDG[p].dot(gridDG[gridIdx]) >= 0:
        offs = ti.atomic_add(activeFieldsCount[gridIdx, 0], 1) #this lets us keep a dynamically sized list by tracking the index
        activeFields[gridIdx, 0, offs] = p #add this particle index to the list!
        grid_m[gridIdx][0] += weight * p_mass #add mass to active field for this particle
        gridSeparability[gridIdx][0] += weight * Dp[p] * p_mass #numerator, field 1
        gridSeparability[gridIdx][2] += weight * p_mass #denom, field 1
        particleAF[p][i*3 + j] = 0 #set this particle's AF to 0 for this grid node
      else:
        offs = ti.atomic_add(activeFieldsCount[gridIdx, 1], 1)
        activeFields[gridIdx, 1, offs] = p
        grid_m[gridIdx][1] += weight * p_mass #add mass to active field for this particle
        gridSeparability[gridIdx][1] += weight * Dp[p] * p_mass #numerator, field 2
        gridSeparability[gridIdx][3] += weight * p_mass #denom, field 2
        particleAF[p][i*3 + j] = 1 #set this particle's AF to 1 for this grid node

  #Iterate grid nodes to compute separability condition and maxDamage (both for each field)
  for i, j in gridSeparability:
    gridIdx = ti.Vector([i, j])

    #Compute seperability for field 1 and store as idx 0
    gridSeparability[gridIdx][0] /= gridSeparability[gridIdx][2] #divide numerator by denominator

    #Compute seperability for field 2 and store as idx 1
    gridSeparability[gridIdx][1] /= gridSeparability[gridIdx][3] #divide numerator by denominator

    #Compute maximum damage for grid node active fields
    max1 = 0.0
    max2 = 0.0
    for k in range(activeFieldsCount[gridIdx, 0]):
      p_i = activeFields[gridIdx, 0, k]
      p_d = Dp[p_i]
      if p_d > max1:
        max1 = p_d
    for k in range(activeFieldsCount[gridIdx, 1]):
      p_i = activeFields[gridIdx, 1, k]
      p_d = Dp[p_i]
      if p_d > max2:
        max2 = p_d
    gridMaxDamage[gridIdx][0] = max1
    gridMaxDamage[gridIdx][1] = max2

    #NOTE: separable[i,j] = -1 for one field, 0 for two non-separable fields, and 1 for two separable fields
    if grid_m[i,j][0] > 0 and grid_m[i,j][1] > 0:
      minSep = gridSeparability[i,j][0] if gridSeparability[i,j][0] < gridSeparability[i,j][1] else gridSeparability[i,j][1]
      maxMax = gridMaxDamage[i,j][0] if gridMaxDamage[i,j][0] > gridMaxDamage[i,j][1] else gridMaxDamage[i,j][1]
      if maxMax == 1.0 and minSep > dMin:
        separable[i,j] = 1
      else:
        separable[i,j] = 0
    
  # Force Update and Velocity Update
  for p in x:
    
    #for particle p, compute base index
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    
    # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]

    mu, la = mu_0, lambda_0

    U, sig, V = ti.svd(F[p])
    J = 1.0

    for d in ti.static(range(2)):
      new_sig = sig[d, d]
      Jp[p] *= sig[d, d] / new_sig
      sig[d, d] = new_sig
      J *= new_sig
    
    #Compute Kirchoff Stress
    kirchoff = kirchoff_FCR(F[p], U@V.transpose(), J, mu, la)

    #P2G for velocity, force update, and update velocity
    for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j])
      gridIdx = base + offset
      dpos = (offset.cast(float) - fx) * dx
      weight = w[i][0] * w[j][1]
      
      dweight = ti.Vector.zero(float,2)
      dweight[0] = inv_dx * dw[i][0] * w[j][1]
      dweight[1] = inv_dx * w[i][0] * dw[j][1]

      force = -p_vol * kirchoff @ dweight

      if separable[gridIdx] == -1: 
        #treat node as one field
        grid_v[gridIdx, 0] += p_mass * weight * (v[p] + C[p] @ dpos) #momentum transfer
        grid_v[gridIdx, 0] += dt * force #add force to update velocity, don't divide by mass bc this is actually updating MOMENTUM
      else:
        #treat node as having two fields
        fieldIdx = int(particleAF[p][i*3 + j]) #grab the field that this particle is in for this node
        grid_v[gridIdx, fieldIdx] += p_mass * weight * (v[p] + C[p] @ dpos) #momentum transfer
        grid_v[gridIdx, fieldIdx] += dt * force #add force to update velocity, don't divide by mass bc this is actually updating MOMENTUM

        grid_n[gridIdx, fieldIdx] += dweight * p_mass #add to the normal for this field at this grid node, remember we need to normalize it later!

  #Frictional Contact Forces
  for i,j in grid_m:
    if separable[i,j] != -1: #only apply these forces to nodes with two fields
      #momentium
      q_1 = grid_v[i, j, 0]
      q_2 = grid_v[i, j, 1]
      q_cm = q_1 + q_2 

      #mass
      m_1 = grid_m[i, j][0]
      m_2 = grid_m[i, j][1]
      m_cm = m_1 + m_2

      #velocity
      v_1 = grid_v[i, j, 0] / m_1
      v_2 = grid_v[i, j, 1] / m_2
      v_cm = q_cm / m_cm #NOTE: we need to compute this like this to conserve mass and momentum

      #normals
      n_1 = grid_n[i, j, 0].normalized() #don't forget to normalize these!!
      n_2 = grid_n[i, j, 1].normalized()
      n_cm1 = (n_1 - n_2).normalized()
      n_cm2 = -n_cm1

      #orthonormal basis for tengent force
      s_cm1 = ti.Vector([-1 * n_cm1[1], n_cm1[0]]) #orthogonal to n_cm
      s_cm2 = s_cm1

      #initialize to hold contact force for each field 
      f_c1 = ti.Vector([0.0, 0.0])
      f_c2 = ti.Vector([0.0, 0.0])

      #Compute these for each field regardless of separable or not
      fNormal1 =  (m_1 / dt) * (v_cm - v_1).dot(n_cm1)
      fNormal2 =  (m_2 / dt) * (v_cm - v_2).dot(n_cm2)
      fTan1 = (m_1 / dt) * (v_cm - v_1).dot(s_cm1)
      fTan2 = (m_2 / dt) * (v_cm - v_2).dot(s_cm2)
      fTanSign1 = 1.0 if fTan1 > 0 else -1.0
      fTanSign2 = 1.0 if fTan2 > 0 else -1.0
      tanDirection1 = (fTan1 * s_cm1).normalized()
      tanDirection2 = (fTan2 * s_cm2).normalized()

      if separable[i,j] == 1:
        #two fields and are separable
        if (v_cm - v_1).dot(n_cm1) > 0:
          tanMin = fricCoeff * abs(fNormal1) if fricCoeff * abs(fNormal1) < abs(fTan1) else abs(fTan1)
          f_c1 += (fNormal1 * n_cm1) + (tanMin * fTanSign1 * tanDirection1)
      
        if (v_cm - v_2).dot(n_cm2) > 0:
          tanMin = fricCoeff * abs(fNormal2) if fricCoeff * abs(fNormal2) < abs(fTan2) else abs(fTan2)
          f_c2 += (fNormal2 * n_cm2) + (tanMin * fTanSign2 * tanDirection2)

      else:
        #two fields but not separable, treat as one field, but each gets an update
        f_c1 += (fNormal1 * n_cm1) + (fTan1 * s_cm1)
        f_c2 += (fNormal2 * n_cm2) + (fTan2 * s_cm2)

      #Now add these forces to our grid node's velocity fields (both of them)
      grid_f[i,j,0] = f_c1
      grid_f[i,j,1] = f_c2
      
      grid_v[i, j, 0] += dt * f_c2 #field 1 gets the contact forces from field 2
      grid_v[i, j, 1] += dt * f_c1 #field 2 gets the contact forces from field 1


  #Gravity and Boundary Collision
  for i, j in grid_m:    
    if separable[i,j] == -1:
      #treat as one field
      nodalMass = grid_m[i,j][0]
      if nodalMass > 0: #if there is mass at this node
        grid_v[i, j, 0] = (1 / nodalMass) * grid_v[i, j, 0] # Momentum to velocity
        
        grid_v[i, j, 0] += dt * gravity[None] # gravity TODO: move to before frictional contact
        
        #add force from mouse
        dist = attractor_pos[None] - dx * ti.Vector([i, j])
        grid_v[i, j, 0] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
        
        #wall collisions
        if i < 3 and grid_v[i, j,0][0] < 0:          grid_v[i, j,0][0] = 0 # Boundary conditions
        if i > n_grid - 3 and grid_v[i, j,0][0] > 0: grid_v[i, j,0][0] = 0
        if j < 3 and grid_v[i, j,0][1] < 0:          grid_v[i, j,0][1] = 0
        if j > n_grid - 3 and grid_v[i, j,0][1] > 0: grid_v[i, j,0][1] = 0

        #hold the top of the box
        if j*dx > 0.575: 
          grid_v[i, j,0][0] = 0
          grid_v[i, j,0][1] = 0

        #move bottom of the box
        if j*dx < 0.42: 
          grid_v[i, j,0][0] = 0
          grid_v[i, j,0][1] = -1
        
    else:
      #treat node as having two fields
      nodalMass1 = grid_m[i,j][0]
      nodalMass2 = grid_m[i,j][1]
      if nodalMass1 > 0 and nodalMass2 > 0: #if there is mass at this node
        grid_v[i, j, 0] = (1 / nodalMass1) * grid_v[i, j, 0] # Momentum to velocity, field 1
        grid_v[i, j, 1] = (1 / nodalMass2) * grid_v[i, j, 1] # Momentum to velocity, field 2
        
        grid_v[i, j, 0] += dt * gravity[None] # gravity, field 1
        grid_v[i, j, 1] += dt * gravity[None] # gravity, field 2
        
        #add force from mouse
        dist = attractor_pos[None] - dx * ti.Vector([i, j])
        grid_v[i, j, 0] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100 #field 1
        grid_v[i, j, 1] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100 #field 2
        
        #wall collisions, field 1
        if i < 3 and grid_v[i, j, 0][0] < 0:          grid_v[i, j, 0][0] = 0 # Boundary conditions
        if i > n_grid - 3 and grid_v[i, j, 0][0] > 0: grid_v[i, j, 0][0] = 0
        if j < 3 and grid_v[i, j,0][1] < 0:          grid_v[i, j, 0][1] = 0
        if j > n_grid - 3 and grid_v[i, j, 0][1] > 0: grid_v[i, j, 0][1] = 0

        #wall collisions, field 2
        if i < 3 and grid_v[i, j, 1][0] < 0:          grid_v[i, j, 1][0] = 0 # Boundary conditions
        if i > n_grid - 3 and grid_v[i, j, 1][0] > 0: grid_v[i, j, 1][0] = 0
        if j < 3 and grid_v[i, j, 1][1] < 0:          grid_v[i, j, 1][1] = 0
        if j > n_grid - 3 and grid_v[i, j, 1][1] > 0: grid_v[i, j, 1][1] = 0
    
  
  # grid to particle (G2P)
  for p in x: 
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
    dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]
    new_v = ti.Vector.zero(float, 2)
    new_C = ti.Matrix.zero(float, 2, 2)
    new_F = ti.Matrix.zero(float, 2, 2)
    for i, j in ti.static(ti.ndrange(3, 3)): # loop over 3x3 grid node neighborhood
      dpos = ti.Vector([i, j]).cast(float) - fx
      gridIdx = base + ti.Vector([i, j])
      g_v = grid_v[gridIdx, 0]
      g_v2 = grid_v[gridIdx, 1] #for field 2
      weight = w[i][0] * w[j][1]

      dweight = ti.Vector.zero(float,2)
      dweight[0] = inv_dx * dw[i][0] * w[j][1]
      dweight[1] = inv_dx * w[i][0] * dw[j][1]

      if separable[gridIdx] == -1:
        #treat as one field
        new_v += weight * g_v
        new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        new_F += g_v.outer_product(dweight)
      else:
        #node has two fields so choose the correct velocity contribution from the node
        fieldIdx = int(particleAF[p][i*3 + j]) #grab the field that this particle is in for this node
        if fieldIdx == 0:
          new_v += weight * g_v
          new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
          new_F += g_v.outer_product(dweight)
        else:
          new_v += weight * g_v2
          new_C += 4 * inv_dx * weight * g_v2.outer_product(dpos)
          new_F += g_v2.outer_product(dweight)

    v[p], C[p] = new_v, new_C
    x[p] += dt * v[p] # advection
    F[p] = (ti.Matrix.identity(float, 2) + (dt * new_F)) @ F[p] #updateF (explicitMPM way)

@ti.kernel
def reset(arr: ti.ext_arr()):

  for i in range(n_particles):
    x[i] = [ti.cast(arr[i,0], ti.f32), ti.cast(arr[i,1], ti.f32)] #cast from f64 to f32 to avoid warnings lol
    #x[i] = ti.Vector([arr[i,0], arr[i,1]])
    #x[i] = [0.4 + ti.random() * 0.2, 0.4 + ti.random() * 0.2]
    material[i] = 0
    v[i] = [0, 0]
    F[i] = ti.Matrix([[1, 0], [0, 1]])
    Jp[i] = 1
    C[i] = ti.Matrix.zero(float, 2, 2)
    Dp[i] = 0
    if (x[i][1] > 0.495) and (x[i][1] < 0.503): #put damaged particles as a band in the center
      Dp[i] = 1
      material[i] = 1
    # for j in range(nSubDivs):
    #   Dp[j] = 1
    #   Dp[particleCount[0] + j]
    #   Dp[particleCount[0] + particleCount[1] + j]
    #   material[j] = 1
    #   material[particleCount[0] + j] = 1
    #   material[particleCount[0] + particleCount[1] + j] = 1

  
print("[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse bottons to attract/repel. Press R to reset.")
gui = ti.GUI("DFG-MPM", res=512, background_color=0xffffff)
reset(vertices)
gravity[None] = [0, 0]
outputPath = "output/testing/brittle.ply"
outputPath2 = "output/testing/brittle_nodes.ply"
fps = 60
endFrame = fps * 10
frameDt = 1.0 / fps
guiOn = False

for frame in range(endFrame):
  if gui.get_event(ti.GUI.PRESS):
    if gui.event.key == 'r': reset(vertices)
    elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: break
  if gui.event is not None: gravity[None] = [0, 0] # if had any event
  if gui.is_pressed(ti.GUI.LEFT,  'a'): gravity[None][0] = -1
  if gui.is_pressed(ti.GUI.RIGHT, 'd'): gravity[None][0] = 1
  if gui.is_pressed(ti.GUI.UP,    'w'): gravity[None][1] = 1
  if gui.is_pressed(ti.GUI.DOWN,  's'): gravity[None][1] = -1
  mouse = gui.get_cursor_pos()
  gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15)
  attractor_pos[None] = [mouse[0], mouse[1]]
  attractor_strength[None] = 0
  if gui.is_pressed(ti.GUI.LMB):
    attractor_strength[None] = 1
  if gui.is_pressed(ti.GUI.RMB):
    attractor_strength[None] = -1
  for s in range(int(frameDt // dt)):
    substep()
  
    numSubsteps = int(frameDt // dt)
    #Write PLY Files
    print('[Simulation]: Writing substep ', s, 'of frame ', frame, '...')
    np_x = x.to_numpy()
    writer = ti.PLYWriter(num_vertices=n_particles)
    writer.add_vertex_pos(np_x[:,0], np_x[:, 1], np.zeros(n_particles)) #add position
    writer.add_vertex_channel("Dp", "double", Dp.to_numpy()) #add damage
    writer.add_vertex_channel("DGx", "double", particleDG.to_numpy()[:,0]) #add particle DG x
    writer.add_vertex_channel("DGy", "double", particleDG.to_numpy()[:,1]) #add particle DG y
    writer.export_frame(frame * numSubsteps + s, outputPath)

    #Construct positions for grid nodes
    gridX = np.zeros((n_grid**2, 2), dtype=float) #format as 1d array of nodal positions
    np_separability = np.zeros(n_grid**2, dtype=int)
    gridNormals = np.zeros((n_grid**2, 4), dtype=float)
    gridFrictionForces = np.zeros((n_grid**2, 4), dtype=float)
    np_DG = np.zeros((n_grid**2, 2), dtype=float)
    for i in range(n_grid):
      for j in range(n_grid):
        gridIdx = i * n_grid + j
        gridX[gridIdx,0] = i * dx
        gridX[gridIdx,1] = j * dx
        np_separability[gridIdx] = separable[i,j] #grab separability
        np_DG[gridIdx, 0] = gridDG[i,j][0]
        np_DG[gridIdx, 1] = gridDG[i,j][1]
        if separable[i,j] != -1:
          gridNormals[gridIdx, 0] = grid_n[i, j, 0][0]
          gridNormals[gridIdx, 1] = grid_n[i, j, 0][1]
          gridNormals[gridIdx, 2] = grid_n[i, j, 1][0]
          gridNormals[gridIdx, 3] = grid_n[i, j, 1][1]
          gridFrictionForces[gridIdx, 0] = grid_f[i, j, 0][0]
          gridFrictionForces[gridIdx, 1] = grid_f[i, j, 0][1]
          gridFrictionForces[gridIdx, 2] = grid_f[i, j, 1][0]
          gridFrictionForces[gridIdx, 3] = grid_f[i, j, 1][1]
    writer2 = ti.PLYWriter(num_vertices=n_grid**2)
    writer2.add_vertex_pos(gridX[:,0], gridX[:, 1], np.zeros(n_grid**2)) #add position
    writer2.add_vertex_channel("sep", "int", np_separability)
    writer2.add_vertex_channel("DGx", "double", np_DG[:,0]) #add particle DG x
    writer2.add_vertex_channel("DGy", "double", np_DG[:,1]) #add particle DG y
    writer2.add_vertex_channel("N_field1_x", "double", gridNormals[:,0]) #add grid_n for field 1 x
    writer2.add_vertex_channel("N_field1_y", "double", gridNormals[:,1]) #add grid_n for field 1 y
    writer2.add_vertex_channel("N_field2_x", "double", gridNormals[:,2]) #add grid_n for field 2 x
    writer2.add_vertex_channel("N_field2_y", "double", gridNormals[:,3]) #add grid_n for field 2 y
    writer2.add_vertex_channel("f_field1_x", "double", gridFrictionForces[:,0]) #add grid_f for field 1 x
    writer2.add_vertex_channel("f_field1_y", "double", gridFrictionForces[:,1]) #add grid_f for field 1 y
    writer2.add_vertex_channel("f_field2_x", "double", gridFrictionForces[:,2]) #add grid_f for field 2 x
    writer2.add_vertex_channel("f_field2_y", "double", gridFrictionForces[:,3]) #add grid_f for field 2 y
    writer2.export_frame(frame * numSubsteps + s, outputPath2)
  
  #Write to Interactive GUI
  if guiOn:
    #colors = np.array([0xED553B,0x068587,0xEEEEF0], dtype=np.uint32)
    colors = np.array([0x000cff,0xff0000,0xEEEEF0], dtype=np.uint32)
    gui.circles(x.to_numpy(), radius=1.5, color=colors[material.to_numpy()])
    for i in range(n_grid+1):
      for j in range(n_grid+1):
        #print(separable[36,39])
        tempX = gridDG[i,j][0]
        tempY = gridDG[i,j][1]
        #print('gridDG[',i,',',j,']:[',tempX,',',tempY,']')
        mag = (tempX**2 + tempY**2)**0.5
        if(mag != 0): 
          if gridMaxDamage[i,j][0] > 0.9:
            gui.arrow([i*dx, j*dx], [tempX * dx * 0.5 / mag, tempY * dx * 0.5/ mag], radius=1, color=0x00ff00)
          else:
            gui.arrow([i*dx, j*dx], [tempX * dx * 0.5 / mag, tempY * dx * 0.5/ mag], radius=1, color=0x000000)
        if separable[i,j] == 1:
          divide = 0.5
          gui.circle([i*dx,j*dx], radius=2, color=0x00ff00)
          n_1 = grid_n[i, j, 0] #don't forget to normalize these!!
          n_1N = [n_1[0] / ((n_1[0]**2 + n_1[1]**2)**0.5), n_1[1] / ((n_1[0]**2 + n_1[1]**2)**0.5) ]
          n_2 = grid_n[i, j, 1]
          n_2N = [n_2[0] / ((n_2[0]**2 + n_2[1]**2)**0.5), n_2[1] / ((n_2[0]**2 + n_2[1]**2)**0.5)]
          n_cm1 = [n_1N[0] - n_2N[0], n_1N[1] - n_2N[1]]
          n_cm1N = [n_cm1[0] / ((n_cm1[0]**2 + n_cm1[1]**2)**0.5), n_cm1[1] / ((n_cm1[0]**2 + n_cm1[1]**2)**0.5)]
          #gui.arrow([i*dx, j*dx], [n_1[0] / divide, n_1[1] /divide], radius=1, color=0x00ff00)
          #gui.arrow([i*dx, j*dx], [-n_1[0] / divide, -n_1[1] / divide], radius=1, color=0x00ff00)
          # gui.arrow([i*dx, j*dx], [n_cm1N[0] / 20.0, n_cm1N[1] /20.0], radius=1, color=0x00ff00)
          # gui.arrow([i*dx, j*dx], [-n_cm1N[0] / 20.0, -n_cm1N[1] / 20.0], radius=1, color=0x00ff00)
        #elif separable[i,j] == -1:
          #gui.circle([i*dx,j*dx], radius=1.5, color=0x000000)
        elif separable[i,j] == 0:
          gui.circle([i*dx,j*dx], radius=2, color=0xf5c842)
  if guiOn: gui.show() # Change to gui.show(f'{frame:06d}.png') to write images to disk