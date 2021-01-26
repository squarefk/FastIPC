import taichi as ti
import math

#-----2D-----#

@ti.func
def eigenDecomposition2D(M):
    e = ti.Vector([0.0, 0.0])
    v1 = ti.Vector([0.0, 0.0])
    v2 = ti.Vector([0.0, 0.0])
    
    x11 = M[0,0]
    x12 = M[0,1]
    x21 = M[1,0]
    x22 = M[1,1]

    if x11 != 0.0 or x12 != 0.0 or x21 != 0.0 or x22 != 0.0: #only go ahead with the computation if M is not all zero
        a = 0.5 * (x11 + x22)
        b = 0.5 * (x11 - x22)
        c = x21

        c_squared = c*c
        m = (b*b + c_squared)**0.5
        k = (x11 * x22) - c_squared

        if a >= 0.0:
            e[0] = a + m
            e[1] = k / e[0] if e[0] != 0.0 else 0.0
        else:
            e[1] = a - m
            e[0] = k / e[1] if e[1] != 0.0 else 0.0

        #exhange sort
        if e[1] > e[0]:
            temp = e[0]
            e[0] = e[1]
            e[1] = temp

        v1 = ti.Vector([m+b, c]).normalized() if b >= 0 else ti.Vector([-c, b-m]).normalized()
        v2 = ti.Vector([-v1[1], v1[0]])

    return e, v1, v2

@ti.kernel
def testEigenDecomp():
    A = ti.Matrix([[0.0, 0.0],[0.0, 0.0]])
    for i in range(10):
        base = -5.0
        interval = 10.0
        a = base + (interval*ti.random())
        b = base + (interval*ti.random())
        c = base + (interval*ti.random())
        A = ti.Matrix([[a,b],[b,c]])
        values, v1, v2 = eigenDecomposition2D(A)
        
        #Now let's reconstruct A
        Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2))
        print("A:", A)
        print("B:", Areconstructed)
        print("e:", values)
        print("v1:", v1)
        print("v2:", v2)
        print()

#-----3D-----#

#3D EigenDecomposition Algorithm from Deledalle2017 on 3x3 Hermitian EigenDecomp: https://hal.archives-ouvertes.fr/hal-01501221/document 
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

        #-----EIGENVALUES-----#

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

        #-----EIGENVECTORS-----#

        #There are four cases here: f = 0; f and d = 0; f and e = 0; f and e and d non zero (the assumed case from the paper)
        #NOTE: I derived the following three case treatments since the paper does not address them

        thresh = 1e-15

        if ti.abs(f) < thresh and ti.abs(d) >= thresh and ti.abs(e) >= thresh:
            #this case looks ugly, bear w me lol
            c2 = e / ((b - values[0]) - (d**2.0 / (a - values[0])))
            c1 = c2 * (-d / (a - values[0]))
            v1[0] = -c1
            v1[1] = -c2
            v1[2] = 1.0
            v1 = v1.normalized()

            c2 = e / ((b - values[1]) - (d**2.0 / (a - values[1])))
            c1 = c2 * (-d / (a - values[1]))
            v2[0] = -c1
            v2[1] = -c2
            v2[2] = 1.0
            v2 = v2.normalized()
            
            c2 = e / ((b - values[2]) - (d**2.0 / (a - values[2])))
            c1 = c2 * (-d / (a - values[2]))
            v3[0] = -c1
            v3[1] = -c2
            v3[2] = 1.0
            v3 = v3.normalized()
            
        elif ti.abs(f) < thresh and ti.abs(d) < thresh and ti.abs(e) >= thresh:

            v1 = ti.Vector([1.0, 0.0, 0.0]) #first eigenvalue corresponds to this (in this case)

            v2[0] = 0.0
            v2[1] = -e / (b - values[1])
            v2[2] = 1.0
            v2 = v2.normalized()

            v3[0] = 0.0
            v3[1] = -e / (b - values[2])
            v3[2] = 1.0
            v3 = v3.normalized()

        elif ti.abs(f) < thresh and ti.abs(d) >= thresh and ti.abs(e) < thresh:
            
            v1[0] = -d / (a - values[0])
            v1[1] = 1.0
            v1[2] = 0.0
            v1 = v1.normalized()

            v2 = ti.Vector([0.0, 0.0, 1.0]) #this one is in the middle for some reason, just make sure it lines up
    
            v3[0] = -d / (a - values[2])
            v3[1] = 1.0
            v3[2] = 0.0
            v3 = v3.normalized()

        else: #f and d and e non-zero case

            #Make sure we aren't dividing by zero, there are four distinct expressions that might be 0: f = 0, and all three denoms of m1, m2, m3
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

#Test eigen decomp with random matrix and all symmetric zero cases
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
    print("v1:", v1)
    print("v2:", v2)
    print("v3:", v3)
    print("")

    # f == 0
    A = ti.Matrix([[a, d, 0.0], [d, b, e], [0.0, e, c]])
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("f == 0")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)
    print("v1:", v1)
    print("v2:", v2)
    print("v3:", v3)
    print("")

    # f == 0 and d == 0
    A = ti.Matrix([[a, 0.0, 0.0], [0.0, b, e], [0.0, e, c]])
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("f == 0 and d == 0")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)
    print("v1:", v1)
    print("v2:", v2)
    print("v3:", v3)
    print("")

    # f == 0 and e == 0
    A = ti.Matrix([[a, d, 0.0], [d, b, 0.0], [0.0, 0.0, c]])
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("f == 0 and e == 0")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)
    print("v1:", v1)
    print("v2:", v2)
    print("v3:", v3)
    print("")

    # f == 0 and d == 0 and e == 0
    A = ti.Matrix([[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]])
    values, v1, v2, v3 = eigenDecomposition3D(A)
    Areconstructed = (values[0] * v1.outer_product(v1)) + (values[1] * v2.outer_product(v2)) + (values[2] * v3.outer_product(v3))
    print("f == 0 and d == 0 and e == 0")
    print("A:", A)
    print("B:", Areconstructed)
    print("vals:", values)
    print("v1:", v1)
    print("v2:", v2)
    print("v3:", v3)
    print("")