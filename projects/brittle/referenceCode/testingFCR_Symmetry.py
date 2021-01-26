import taichi as ti
import numpy as np
import triangle as tr
import math
from projects.brittle.utils.eigenDecomposition import *

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel

@ti.kernel
def main():
    E = 1e4
    nu = 0.15
    mu = E / (2 * (1 + nu))
    la = E * nu / ((1+nu) * (1 - 2 * nu))

    c1 = -1e-12 #e-9 seems to be a good threshold for treating these as zeros! Since it still works for this value
    c2 = -2*c1

    F = ti.Matrix([[1.0, 0.0, 0.0], [c1, 1.0, c2], [0.0, 0.0, 1.0]])
    U, sig, V = ti.svd(F)
    U_det = U.determinant()
    sig_det = sig.determinant()
    V_det = V.determinant()
    J_sig = sig[0,0] * sig[1,1] * sig[2,2]
    R = U@V.transpose()

    k1 = 2 * mu * (F - R) @ F.transpose()
    k2 = ti.Matrix.identity(float, 3) * la * J_sig * (J_sig - 1)
    kirchoff = k1 + k2 #compute kirchoff stress for FCR model (remember tau = P F^T)

    presetCauchy = ti.Matrix([[0.0, -1.0, 0.0], [-1.0, 0.0, 20.0], [0.0, 20.0, 0.0]])

    #e, v1, v2, v3 = eigenDecomposition3D(kirchoff / J_sig)
    e, v1, v2, v3 = eigenDecomposition3D(presetCauchy)

    scale = 1.0

    print("F:",F)
    print("U:", U)
    print("V:", V)
    print("sig:", sig)
    print("J_sig:", J_sig*scale)
    print("U_det:", U_det*scale)
    print("sig_det:", sig_det*scale)
    print("V_det:", V_det*scale)
    print("R:", R)
    print("k1:", k1)
    print("k2:", k2)
    print("kirchoff:", kirchoff)
    print("e:", e)
    print("v1:", v1)
    print("v2:", v2)
    print("v3:", v3)

# main()
testEigenDecomp3D()
