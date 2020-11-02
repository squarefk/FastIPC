import taichi as ti
import numpy as np

ti.init(arch=ti.gpu) # Try to run on GPU

quality = 1 # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality ** 2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 1e3, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters

x = ti.Vector.field(2, dtype=float, shape=n_particles) # position
v = ti.Vector.field(2, dtype=float, shape=n_particles) # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles) # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles) # deformation gradient
material = ti.field(dtype=int, shape=n_particles) # material id
Jp = ti.field(dtype=float, shape=n_particles) # plastic deformation
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid)) # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid)) # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())

@ti.func 
def kirchoff_FCR(F, R, J, mu, la):
  return 2 * mu * (F - R) @ F.transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1) #compute kirchoff stress for FCR model (remember tau = P F^T)

@ti.kernel
def substep():
  #re-initialize grid quantities
  for i, j in grid_m:
    grid_v[i, j] = [0, 0]
    grid_m[i, j] = 0

  # Particle state update and scatter to grid (P2G)
  for p in x: 
    
    #for particle p, compute base index
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    
    # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]

    #F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p] #updateF (mlsmpm way)

    #h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p])))) # Hardening coefficient: snow gets harder when compressed
    
    # if material[p] == 0: # jelly, make it softer
    #   h = 0.3

    mu, la = mu_0, lambda_0
    #mu, la = mu_0 * h, lambda_0 * h

    # if material[p] == 1: # liquid
    #   mu = 0.0

    U, sig, V = ti.svd(F[p])
    J = 1.0

    for d in ti.static(range(2)):
      new_sig = sig[d, d]
      # if material[p] == 2:  # Snow
      #   new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
      Jp[p] *= sig[d, d] / new_sig
      sig[d, d] = new_sig
      J *= new_sig

    # if material[p] == 1:  # Reset deformation gradient to avoid numerical instability
    #   F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)

    # elif material[p] == 2:
    #   F[p] = U @ sig @ V.transpose() # Reconstruct elastic deformation gradient after plasticity
    
    #Compute Kirchoff Stress
    kirchoff = kirchoff_FCR(F[p], U@V.transpose(), J, mu, la)

    #P2G for velocity and mass AND Force Update!
    for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j])
      dpos = (offset.cast(float) - fx) * dx
      weight = w[i][0] * w[j][1]
      
      dweight = ti.Vector.zero(float,2)
      dweight[0] = inv_dx * dw[i][0] * w[j][1]
      dweight[1] = inv_dx * w[i][0] * dw[j][1]
      
      #grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos) #mlsmpm force update
      force = -p_vol * kirchoff @ dweight

      grid_v[base + offset] += p_mass * weight * (v[p] + C[p] @ dpos) #momentum transfer
      grid_m[base + offset] += weight * p_mass #mass transfer

      grid_v[base + offset] += dt * force #add force to update velocity, don't divide by mass bc this is actually updating MOMENTUM
  
  # Gravity and Boundary Collision
  for i, j in grid_m:
    if grid_m[i, j] > 0: # No need for epsilon here
      grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j] # Momentum to velocity
      
      grid_v[i, j] += dt * gravity[None] * 30 # gravity
      
      #add force from mouse
      dist = attractor_pos[None] - dx * ti.Vector([i, j])
      grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
      
      #wall collisions
      if i < 3 and grid_v[i, j][0] < 0:          grid_v[i, j][0] = 0 # Boundary conditions
      if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
      if j < 3 and grid_v[i, j][1] < 0:          grid_v[i, j][1] = 0
      if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0
  
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
      g_v = grid_v[base + ti.Vector([i, j])]
      weight = w[i][0] * w[j][1]

      dweight = ti.Vector.zero(float,2)
      dweight[0] = inv_dx * dw[i][0] * w[j][1]
      dweight[1] = inv_dx * w[i][0] * dw[j][1]

      new_v += weight * g_v
      new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
      new_F += g_v.outer_product(dweight)
    v[p], C[p] = new_v, new_C
    x[p] += dt * v[p] # advection
    F[p] = (ti.Matrix.identity(float, 2) + (dt * new_F)) @ F[p] #updateF (explicitMPM way)

@ti.kernel
def reset():
  group_size = n_particles // 1
  for i in range(n_particles):
    x[i] = [ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size), ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)]
    material[i] = i // group_size # 0: fluid 1: jelly 2: snow
    v[i] = [0, 0]
    F[i] = ti.Matrix([[1, 0], [0, 1]])
    Jp[i] = 1
    C[i] = ti.Matrix.zero(float, 2, 2)
  
print("[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse bottons to attract/repel. Press R to reset.")
gui = ti.GUI("Explicit MPM", res=512, background_color=0x112F41)
reset()
gravity[None] = [0, 0]

for frame in range(20000):
  if gui.get_event(ti.GUI.PRESS):
    if gui.event.key == 'r': reset()
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
  for s in range(int(2e-3 // dt)):
    substep()
  colors = np.array([0xED553B,0x068587,0xEEEEF0], dtype=np.uint32)
  gui.circles(x.to_numpy(), radius=1.5, color=colors[material.to_numpy()])
  gui.show() # Change to gui.show(f'{frame:06d}.png') to write images to disk