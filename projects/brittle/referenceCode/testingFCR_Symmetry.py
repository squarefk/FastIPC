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

        #Now let's threshold d, e, and f to make sure they are not just numerical error
        epsilon = 1e-40
        if d == 0: d = epsilon
        if e == 0: e = epsilon
        if f == 0: f = epsilon

        # scale = 1
        # print("d:", d*scale, "e:", e*scale, "f:", f*scale)

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

        #Make sure we aren't dividing by zero, there are four distinct expressions that might be 0: f = 0, and all three denoms of m1, m2, m3
        if f == 0:
            print("[EigenDecomp] ERROR: f == 0 with M:", M*1e10, "location: ", location)
        if ((f*(b - values[0])) - (d*e)) == 0:
            print("[EigenDecomp] ERROR: ((f*(b - values[0])) - (d*e)) == 0 with M:", M*1e10, "location: ", location)
        if ((f*(b - values[1])) - (d*e)) == 0:
            print("[EigenDecomp] ERROR: ((f*(b - values[1])) - (d*e)) == 0 with M:", M*1e10, "location: ", location)
        if ((f*(b - values[2])) - (d*e)) == 0:
            print("[EigenDecomp] ERROR: ((f*(b - values[2])) - (d*e)) == 0 with M:", M*1e10, "location: ", location)

        #And now we must compute the eigenvectors
        m1 = ((d * (c - values[0])) - (e*f)) / ((f*(b - values[0])) - (d*e))
        m2 = ((d * (c - values[1])) - (e*f)) / ((f*(b - values[1])) - (d*e))
        m3 = ((d * (c - values[2])) - (e*f)) / ((f*(b - values[2])) - (d*e))

        v1[0] = (values[0] - c - (e*m1)) / f
        v1[1] = m1
        v1[2] = 1.0
        v1 = v1.normalized()

        v2[0] = (values[1] - c - (e*m2)) / f
        v2[1] = m2
        v2[2] = 1.0
        v2 = v2.normalized()

        v3[0] = (values[2] - c - (e*m3)) / f
        v3[1] = m3
        v3[2] = 1.0
        v3 = v3.normalized()
    
    #Reorder so that our eigenvalues are in descending order, simple bubble sort bc n=3 lol
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
    print("")

    # f == 0
    A = ti.Matrix([[a, d, 0.0], [d, b, e], [0.0, e, c]])
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("f == 0")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)
    print("")

    # f == 0 and d == 0
    A = ti.Matrix([[a, 0.0, 0.0], [0.0, b, e], [0.0, e, c]])
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("f == 0 and d == 0")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)
    print("")

    # f == 0 and e == 0
    A = ti.Matrix([[a, d, 0.0], [d, b, 0.0], [0.0, 0.0, c]])
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("f == 0 and e == 0")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)
    print("")

    # f == 0 and d == 0 and e == 0
    A = ti.Matrix([[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]])
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("f == 0 and d == 0 and e == 0")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)
    print("")

#main()
testEigenDecomp3D()
