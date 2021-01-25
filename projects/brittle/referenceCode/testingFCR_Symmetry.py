import taichi as ti
import numpy as np
import triangle as tr
import math

ti.init(default_fp=ti.f64, arch=ti.gpu) # Try to run on GPU    #GPU, parallel

@ti.func
def eigenDecomposition3D(M, location=0):
    values = ti.Vector([0.0, 0.0, 0.0])
    v1 = ti.Vector([0.0, 0.0, 0.0])
    v2 = ti.Vector([0.0, 0.0, 0.0])
    v3 = ti.Vector([0.0, 0.0, 0.0])

    #NOTE: eigenvalue computation uses lower half of matrix, vector computation uses upper half (they should be identical, but numerical error can throw this off)
    a = M[0,0]
    b = M[1,1]
    c = M[2,2]
    d = M[1,0]
    e = M[2,1]
    f = M[2,0]

    #First check if we have a diagonal matrix
    if d**2 + e**2 + f**2 == 0.0:
        values[0] = a
        values[1] = b
        values[2] = c
        v1 = ti.Vector([1.0, 0.0, 0.0])
        v2 = ti.Vector([0.0, 1.0, 0.0])
        v3 = ti.Vector([0.0, 0.0, 1.0])
    else:

        #First we compute eigenvalues using the routine here: Deledalle2017 on 3x3 Hermitian EigenDecomp: https://hal.archives-ouvertes.fr/hal-01501221/document 
        x1 = a**2 + b**2 + c**2 - (a*b) - (a*c) - (b*c) + 3*(d**2 + f**2 + e**2)
        x2 = -1*(2*a - b - c) * (2*b - a - c) * (2*c - a - b)
        x2 += 9 * (((2*c - a - b) * d**2) + ((2*b - a - c) * f**2) + ((2*a - b - c) * e**2))
        x2 -= 54 * (d*e*f)

        phi = math.pi / 2.0
        if x2 > 0:
            phi = ti.atan2(ti.sqrt(4 * x1**3 - x2**2), x2) #atan2(y,x) need to feed numerator and denominator
            #print("x2 > 0")
        elif x2 < 0:
            phi = ti.atan2(ti.sqrt(4 * x1**3 - x2**2), x2) #NOTE: don't add the pi here bc use atan2, already in the proper quadrant 
            #print("x2 < 0")

        values[0] = (a + b + c - (2 * ti.sqrt(x1) * ti.cos(phi / 3.0))) / 3.0
        values[1] = (a + b + c + (2 * ti.sqrt(x1) * ti.cos((phi - math.pi)/ 3.0))) / 3.0
        values[2] = (a + b + c + (2 * ti.sqrt(x1) * ti.cos((phi + math.pi)/ 3.0))) / 3.0

        #----EIGENVECTORS----

        #Now we must compute the eigenvectors, and to do this we refer to this work: Kopp2008 https://www.mpi-hd.mpg.de/personalhomes/globes/3x3/index.html 
        wmax = ti.abs(values[0])
        wmax = ti.abs(values[1]) if wmax < ti.abs(values[1]) else wmax
        wmax = ti.abs(values[2]) if wmax < ti.abs(values[2]) else wmax
        doubleEpsilon = np.finfo(np.float64).eps
        thresh = (8.0 * doubleEpsilon * wmax)**2.0 #this is used as a threshold for floating point comparisons

        #prepare for calculating eigenvectors
        n0tmp = d**2.0 + f**2.0 #temps that will save some flops
        n1tmp = d**2.0 + e**2.0
        #temp storage in v2
        v2[0] = d*e - f*b
        v2[1] = f*d - e*a
        v2[2] = d**2.0

        #Calculate first eigenvector using v[0] = (A - w[0]).e1 x (A - w[0]).e2
        a -= values[0]
        b -= values[0]
        v1[0] = v2[0] + f*values[0]
        v1[1] = v2[1] + e*values[0]
        v1[2] = a*b - v2[2]
        norm = v1[0]**2.0 + v1[1]**2.0 + v1[2]**2.0
        n0 = n0tmp + a**2.0
        n1 = n1tmp + b**2.0
        error = n0 * n1

        if n0 <= thresh: #if first column is zero, then (1,0,0) is an eigenvector
            v1 = ti.Vector([1.0, 0.0, 0.0])
        elif n1 <= thresh: #if second col is zero, then (0,1,0) is eigenvector
            v1 = ti.Vector([0.0, 1.0, 0.0])
        elif norm < ((64.0 * doubleEpsilon)**2.0 * error): #If angle between A[0] and A[1] is too small (not linearly independent), don't use cross product, but calculate v ~ (1, -A0/A1, 0)
            t = d**2.0
            g = -a / d
            if b**2.0 > t:
                t = b**2.0
                g = -d / b 
            if e**2.0 > t:
                g = -f / e
            norm = 1.0 / ti.sqrt(1 + g**2.0)
            v1 = ti.Vector([norm, g * norm, 0.0]) # (1, -A0/A1, 0)
        else: #standard branch, normalize v1
            norm = ti.sqrt(1.0 / norm)
            v1 *= norm

        #Second eigenvector
        t = values[0] - values[1]
        if ti.abs(t) > (8.0 * doubleEpsilon * wmax): #non-degenerate eigenvalues
            # For non-degenerate eigenvalue, calculate second eigenvector by the formula v[1] = (A - w[1]).e1 x (A - w[1]).e2
            a += t #this adds val[0] back in and subtracts out val[1], clever!
            b += t 
            v2[0] = v2[0] + f*values[1]
            v2[1] = v2[1] + e*values[1]
            v2[2] = a*b - v2[2]
            norm = v2[0]**2.0 + v2[1]**2.0 + v2[2]**2.0
            n0 = n0tmp + a**2.0
            n1 = n1tmp + b**2.0
            error = n0 * n1

            if n0 <= thresh: #if first column is zero, then (1,0,0) is an eigenvector
                v2 = ti.Vector([1.0, 0.0, 0.0])
            elif n1 <= thresh: #if second col is zero, then (0,1,0) is eigenvector
                v2 = ti.Vector([0.0, 1.0, 0.0])
            elif norm < ((64.0 * doubleEpsilon)**2.0 * error): #If angle between A[0] and A[1] is too small (not linearly independent), don't use cross product, but calculate v ~ (1, -A0/A1, 0)
                t = d**2.0
                g = -a / d
                if b**2.0 > t:
                    t = b**2.0
                    g = -d / b 
                if e**2.0 > t:
                    g = -f / e
                norm = 1.0 / ti.sqrt(1 + g**2.0)
                v2 = ti.Vector([norm, g * norm, 0.0]) # (1, -A0/A1, 0)
            else: #standard branch, normalize v2
                norm = ti.sqrt(1.0 / norm)
                v2 *= norm
        else: #val[0] == val[1] (degenerate eigenvals)
            #For degenerate eigenvalue, calculate second eigenvector according to v[1] = v[0] x (A - w[1]).e[i]
            #ensure that the matrix is fully diagonal for this routine NOTE: this copies lower left triangular into upper right triangular, guaranteeing that it's diagonal
            M[0,1] = d
            M[1,2] = e
            M[0,2] = f
            # broken = False
            # for i in range(3):
            #     if broken == False:
            #         M[i][i] = M[i][i] - values[1]
            #         n0 = M[0][i]**2.0 + M[1][i]**2.0 + M[2][i]**2.0
            #         if n0 > thresh:
            #             v2[0] = v1[1]*M[2][i] - v1[2]*M[1][i]
            #             v2[1] = v1[2]*M[0][i] - v1[0]*M[2][i]
            #             v2[2] = v1[0]*M[1][i] - v1[1]*M[0][i]
            #             norm = v2[0]**2.0 + v2[1]**2.0 + v2[2]**2.0
            #             if norm > ((256.0 * doubleEpsilon)**2.0 * n0): 
            #                 # Accept cross product only if the angle between the two vectors was not too small
            #                 norm = ti.sqrt(1.0 / norm)
            #                 v2 *= norm
            #                 broken = True
            # if broken == False: #means we exited without breaking at all, so any orthogonal vector to v[0] is an EV!
            #     #find first nonzero of v1
            #     for j in range(3):
            #         if v1[j] != 0 and broken == False: 
            #             norm = 1.0 / ti.sqrt(v1[j]**2.0 + v1[(j+1)%3]**2.0)
            #             v2[j] = v1[(j+1)%3] * norm
            #             v2[(j+1)%3] = -v1[j] * norm
            #             v2[(j+2)%3] = 0.0
            #             broken = True

        #Eigenvector three we can easily compute according to v[2] = v[0] x v[1]
        v3[0] = v1[1] * v2[2] - v1[2] * v2[1]
        v3[1] = v1[2] * v2[0] - v1[0] * v2[2]
        v3[2] = v1[0] * v2[1] - v1[1] * v2[0]

    #Reorder so that our eigenvalues are in descending order, simple bubble sort bc n=3
    sorted = False
    while sorted == False:
        sorted = True
        if values[0] < values[1]:
            #swap
            temp = values[0]
            values[0] = values[1]
            values[1] = temp
            temp2 = v1
            v1 = v2
            v2 = temp2
            sorted = False
        if values[1] < values[2]:
            #swap
            temp = values[1]
            values[1] = values[2]
            values[2] = temp
            temp2 = v2
            v2 = v3
            v3 = temp2
            sorted = False

    return values, v1, v2, v3

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

@ti.kernel
def testEigenDecomp3D():
    A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    base = -5.0
    interval = 10.0
    a = base + (interval*ti.random())
    b = base + (interval*ti.random())
    c = base + (interval*ti.random())
    d = base + (interval*ti.random())
    e = base + (interval*ti.random())
    f = base + (interval*ti.random())
    
    #RANDOM MATRIX
    A = ti.Matrix([[a, d, f], [d, b, e], [f, e, c]]) #ensure matrix is symmetric since it will be a Cauchy stress
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("Random Matrix")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)

    # f == 0
    A = ti.Matrix([[a, d, 0.0], [d, b, e], [0.0, e, c]])
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("f == 0")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)

    # f == 0 and d == 0
    A = ti.Matrix([[a, 0.0, 0.0], [0.0, b, e], [0.0, e, c]])
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("f == 0 and d == 0")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)

    # f == 0 and e == 0
    A = ti.Matrix([[a, d, 0.0], [d, b, 0.0], [0.0, 0.0, c]])
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("f == 0 and e == 0")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)

    # f == 0 and d == 0 and e == 0
    A = ti.Matrix([[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]])
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("f == 0 and d == 0 and e == 0")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)

#main()
testEigenDecomp3D()
