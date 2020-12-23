import taichi as ti

@ti.func
def dihedral_angle(v0, v1, v2, v3):
    n1 = ti.cross(v1 - v0, v2 - v0)
    n2 = ti.cross(v2 - v3, v1 - v3)
    DA = ti.acos(ti.max(T(-1), ti.min(T(1), ti.dot(n1, n2) / ti.sqrt(n1.norm_sqr() * n2.norm_sqr()))))
    if ti.dot(ti.cross(n2, n1), v1 - v2) < 0:
        DA = -DA
    return DA

@ti.func
def dihedral_angle_gradient(v2, v0, v1, v3): 
    # here we map our v order to rusmas' in this function for implementation convenience
    e0 = v1 - v0
    e1 = v2 - v0
    e2 = v3 - v0
    e3 = v2 - v1
    e4 = v3 - v1
    n1 = ti.cross(e0, e1)
    n2 = ti.cross(e2, e0)
    n1SqNorm = n1.norm_sqr()
    n2SqNorm = n2.norm_sqr()
    e0norm = e0.norm()
    da_dv2 = -e0norm / n1SqNorm * n1
    da_dv0 = -e0.dot(e3) / (e0norm * n1SqNorm) * n1 - e0.dot(e4) / (e0norm * n2SqNorm) * n2
    da_dv1 = e0.dot(e1) / (e0norm * n1SqNorm) * n1 + e0.dot(e2) / (e0norm * n2SqNorm) * n2
    da_dv3 = -e0norm / n2SqNorm * n2
    grad = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for d in ti.static(range(3)):
        grad[0 * 3 + d] = da_dv2[d]
        grad[1 * 3 + d] = da_dv0[d]
        grad[2 * 3 + d] = da_dv1[d]
        grad[3 * 3 + d] = da_dv3[d]
    return grad


@ti.func
def compute_mHat(xp, xe0, xe1):
    e = xe1 - xe0
    mHat = xe0 + ti.dot(xp - xe0, e) / e.norm_sqr() * e - xe1
    mHat /= mHat.norm()
    return mHat

@ti.func
def dihedral_angle_hessian(v2, v0, v1, v3):
    e = (v1 - v0, v2 - v0, v3 - v0, v2 - v1, v3 - v1)
    norm_e = (e[0].norm(), e[1].norm(), e[2].norm(), e[3].norm(), e[4].norm())
    n1 = ti.cross(e[0], e[1])
    n2 = ti.cross(e[2], e[0])
    n1norm = n1.norm()
    n2norm = n2.norm()
    mHat1, mHat2, mHat3, mHat4, mHat01, mHat02 = compute_mHat(v1, v0, v2), compute_mHat(v1, v0, v3), compute_mHat(v0, v1, v2), compute_mHat(v0, v1, v3), compute_mHat(v2, v0, v1), compute_mHat(v3, v0, v1)
    cosalpha1 = ti.dot(e[0], e[1]) / (norm_e[0] * norm_e[1])
    cosalpha2 = ti.dot(e[0], e[2]) / (norm_e[0] * norm_e[2])
    cosalpha3 = - ti.dot(e[0], e[3]) / (norm_e[0] * norm_e[3])
    cosalpha4 = - ti.dot(e[0], e[4]) / (norm_e[0] * norm_e[4])
    h1 = n1norm / norm_e[1]
    h2 = n2norm / norm_e[2]
    h3 = n1norm / norm_e[3]
    h4 = n2norm / norm_e[4]
    h01 = n1norm / norm_e[0]
    h02 = n2norm / norm_e[0]

    N1_01 = n1 @ (mHat01.transpose() / (h01 * h01 * n1norm))
    N1_3 = n1 @ (mHat3.transpose() / (h01 * h3 * n1norm))
    N1_1 = n1 @ (mHat1.transpose() / (h01 * h1 * n1norm))
    N2_4 = n2 @ (mHat4.transpose() / (h02 * h4 * n2norm))
    N2_2 = n2 @ (mHat2.transpose() / (h02 * h2 * n2norm))
    N2_02 = n2 @ (mHat02.transpose() / (h02 * h02 * n2norm))
    M3_01_1 = (cosalpha3 / (h3 * h01 * n1norm) * mHat01) @ n1.transpose()
    M1_01_1 = (cosalpha1 / (h1 * h01 * n1norm) * mHat01) @ n1.transpose()
    M1_1_1 = (cosalpha1 / (h1 * h1 * n1norm) * mHat1) @ n1.transpose()
    M3_3_1 = (cosalpha3 / (h3 * h3 * n1norm) * mHat3) @ n1.transpose()
    M3_1_1 = (cosalpha3 / (h3 * h1 * n1norm) * mHat1) @ n1.transpose()
    M1_3_1 = (cosalpha1 / (h1 * h3 * n1norm) * mHat3) @ n1.transpose()
    M4_02_2 = (cosalpha4 / (h4 * h02 * n2norm) * mHat02) @ n2.transpose()
    M2_02_2 = (cosalpha2 / (h2 * h02 * n2norm) * mHat02) @ n2.transpose()
    M4_4_2 = (cosalpha4 / (h4 * h4 * n2norm) * mHat4) @ n2.transpose()
    M2_4_2 = (cosalpha2 / (h2 * h4 * n2norm) * mHat4) @ n2.transpose()
    M4_2_2 = (cosalpha4 / (h4 * h2 * n2norm) * mHat2) @ n2.transpose()
    M2_2_2 = (cosalpha2 / (h2 * h2 * n2norm) * mHat2) @ n2.transpose()
    B1 = n1 * (mHat01.transpose() / (norm_e[0] * norm_e[0] * n1norm))
    B2 = n2 * (mHat02.transpose() / (norm_e[0] * norm_e[0] * n2norm))

    Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hess = ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z])
    H00 = -(N1_01 + N1_01.transpose())
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            ii, jj = i, j
            hess[ii, jj] = H00[i, j]

    H10 = M3_01_1 - N1_3
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            ii, jj = 3 + i, j
            hess[ii, jj] = H10[i, j]
            hess[jj, ii] = H10[i, j]
             
    H20 = M1_01_1 - N1_1
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            ii, jj = 6 + i, j
            hess[ii, jj] = H20[i, j]
            hess[jj, ii] = H20[i, j]
    
    #H30 is zero

    H11 = M3_3_1 + M3_3_1.transpose() - B1 + M4_4_2 + M4_4_2.transpose() - B2
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            ii, jj = 3 + i, 3 + j
            hess[ii, jj] = H11[i, j]

    H12 = M3_1_1 + M1_3_1.transpose() + B1 + M4_2_2 + M2_4_2.transpose() + B2
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            ii, jj = 3 + i, 6 + j
            hess[ii, jj] = H12[i, j]
            hess[jj, ii] = H12[i, j]

    H13 = M4_02_2 - N2_4
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            ii, jj = 3 + i, 9 + j
            hess[ii, jj] = H13[i, j]
            hess[jj, ii] = H13[i, j]

    H22 = M1_1_1 + M1_1_1.transpose() - B1 + M2_2_2 + M2_2_2.transpose() - B2
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            ii, jj = 6 + i, 6 + j
            hess[ii, jj] = H22[i, j]

    H23 = M2_02_2 - N2_2
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            ii, jj = 6 + i, 9 + j
            hess[ii, jj] = H23[i, j]
            hess[jj, ii] = H23[i, j]

    H33 = -(N2_02 + N2_02.transpose())
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            ii, jj = 9 + i, 9 + j
            hess[ii, jj] = H33[i, j]
    return hess