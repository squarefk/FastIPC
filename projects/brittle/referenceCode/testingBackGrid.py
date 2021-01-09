import taichi as ti
import numpy as np
import triangle as tr

#ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

#Sparse Grids
#---Params
dim = 3
dx = 0.003
rp = (3*(dx**2))**0.5
invDx = 1.0 / 0.003
nGrid = ti.ceil(invDx)
grid_size = 4096
grid_block_size = 128
leaf_block_size = 16 if dim == 2 else 8
indices = ti.ij if dim == 2 else ti.ijk
offset = tuple(0 for _ in range(dim))
#---Grid Shapes for PID
grid = ti.root.pointer(indices, grid_size // grid_block_size) # 32
block = grid.pointer(indices, grid_block_size // leaf_block_size) # 8
pid = ti.field(int)
block.dynamic(ti.indices(dim), 1024 * 1024, chunk_size=leaf_block_size**dim * 8).place(pid, offset=offset + (0, ))
#---Grid Shapes for Rest of Grid Structures
grid2 = ti.root.pointer(indices, grid_size // grid_block_size) # 32
block2 = grid2.pointer(indices, grid_block_size // leaf_block_size) # 8
def block_component(c):
    block2.dense(indices, leaf_block_size).place(c, offset=offset) # 16 in 3D, 8 in 2D (-2048, 2048) or (0, 4096) w/o offset

#Grid Structures
gridNumParticles = ti.field(dtype=int)      #track number of particles in each cell using cell index
maxPPC = 2**10
block_component(gridNumParticles) #keep track of how many particles are at each cell of backGrid
backGrid = ti.field(int)              #background grid to map grid cells to a list of particles they contain
backGridIndeces = ti.ijk if dim == 2 else ti.ijkl
backGridShape = (nGrid, nGrid, maxPPC) if dim == 2 else (nGrid, nGrid, nGrid, maxPPC)
ti.root.dense(backGridIndeces, backGridShape).place(backGrid)      #backGrid is nGrid x nGrid x maxPPC

#Particle Structures
x = ti.Vector.field(dim, dtype=float) # position
mp = ti.field(dtype=float) # particle masses
particle = ti.root.dynamic(ti.i, 2**27, 2**19) #2**20 causes problems in CUDA (maybe asking for too much space)
particle.place(x, mp)

#Neighbor Search Routines
@ti.func
def backGridIdx(x):
    #compute int vector of backGrid indeces (recall the grid here )
    return int(x/rp)

@ti.func
def stencil_range():
    return ti.ndrange(*((3, ) * dim))

def addParticles():
    N = 10
    w = 0.2
    dw = w / N
    for i in range(N):
        for j in range(N):
            for k in range(N):
                x[i*(N**2) + j*N + k] = [0.4 + (i * dw), 0.4 + (j * dw), 0.4 + (k * dw)]
                mp[i*(N**2) + j*N + k] = 0.001
            
@ti.kernel
def build_pid():
    ti.block_dim(64) #this sets the number of threads per block / block dimension
    for p in x:
        base = int(ti.floor(x[p] * invDx - 0.5))
        ti.append(pid.parent(), base - ti.Vector(list(offset)), p)

@ti.kernel
def backGridSort():
    #Sort particles into backGrid
    ti.block_dim(256)
    ti.no_activate(particle)
    for I in ti.grouped(pid):
        p = pid[I]
        cell = backGridIdx(x[p]) #grab cell idx (vector of ints)
        offs = ti.atomic_add(gridNumParticles[cell], 1) #atomically add one to our grid cell's particle count NOTE: returns the OLD value before add
        print("cell:", cell, "offs:", offs)
        print("backGrid shape:", backGrid.shape)
        backGrid[cell, offs] = p #place particle idx into the grid cell bucket at the correct place in the cell's neighbor list (using offs)
 

addParticles()
grid.deactivate_all()
build_pid()
backGridSort()


# elements = []
# for i in range(n):
#     elements.append(x[i])
# print(elements)
