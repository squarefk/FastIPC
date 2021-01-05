import taichi as ti
import numpy as np
import triangle as tr

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel
#ti.init(default_fp=ti.f64, arch=ti.cpu, cpu_max_num_threads=1)  #CPU, sequential

@ti.func
def computeB(Yi, ci, chat):
    c = ci / chat
    return -Yi * (c - 1)**2 * ti.log(c) if ci < chat else 0.0

@ti.func
def computeBPrime(Yi, ci, chat):
    c = ci / chat
    return -Yi * ((2.0 * (c - 1.0) * ti.log(c) / chat) + ((c - 1.0)**2 / ci)) if ci < chat else 0.0

@ti.func
def computeBDoublePrime(Yi, ci, chat):
    c = ci / chat
    return -Yi * (((2.0 * ti.log(c) + 3.0) / chat**2) - (2.0 / (ci * chat)) - (1 / ci**2)) if ci < chat else 0.0

@ti.func
def computeB_Minchen(Y, c, chat):
    scaled_c = c/chat
    return -Y * (scaled_c - 1)**2 * ti.log(scaled_c) if c < chat else 0.0

@ti.func
def computeBPrime_Minchen(Y, c, chat):
    scaled_c = c/chat
    return Y * (-2 * (scaled_c - 1) * ti.log(scaled_c) / chat - (scaled_c - 1)**2 / c) if c < chat else 0.0

@ti.func
def computeBDoublePrime_Minchen(Y, c, chat):
    scaled_c = c/chat
    return Y * ((-2 * ti.log(scaled_c) - 3) / chat**2 + 2 / (c * chat) + 1 / (c * c)) if c < chat else 0.0

@ti.kernel
def main():
    Y = 123
    ci = 0.2
    chat = 1.0
    print("computeB Josh:", computeB(Y, ci, chat))
    print("computeBPrime Josh:", computeBPrime(Y, ci, chat))
    print("computeBDoublePrime Josh:", computeBDoublePrime(Y, ci, chat))
    print("computeB Minchen:", computeB_Minchen(Y, ci, chat))
    print("computeBPrime Minchen:", computeBPrime_Minchen(Y, ci, chat))
    print("computeBDoublePrime Minchen:", computeBDoublePrime_Minchen(Y, ci, chat))
    
main()
