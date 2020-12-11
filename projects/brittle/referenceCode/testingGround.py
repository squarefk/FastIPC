import taichi as ti
import numpy as np
import triangle as tr

ti.init(arch=ti.gpu) # Try to run on GPU
#ti.init(arch=ti.cpu, cpu_max_num_threads=1)


# x = ti.field(ti.i32)
# gridSize = 3 
# n = 5

# gridShape = ti.root.dense(ti.ij, (gridSize,gridSize)) #gridSize x gridSize shape

# gridShape.dynamic(ti.i, n, 32).place(x) #add dynamic list to each grid node element

@ti.kernel
def test():
    I = ti.Matrix.identity(float, 2)
    print('contraction:', I*I) #NOTE: this turns out to NOT be contraction lol
    # for i in range(n):
    #     ti.append(x.parent(), [], i)

test()

# elements = []
# for i in range(n):
#     elements.append(x[i])
# print(elements)
