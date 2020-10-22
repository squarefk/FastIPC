import taichi as ti
from common.math.math_tools import *


@ti.func
def elasticity_energy(sig: ti.template(), la, mu):
    sigma = ti.Matrix.zero(ti.get_runtime().default_fp, sig.n, 1)
    for i in ti.static(range(sig.n)):
        sigma[i] = sig[i, 0 if ti.static(sig.m == 1) else i]
    sigmam12Sum = (sigma - ti.Vector([1, 1])).norm_sqr()
    sigmaProdm1 = sigma[0] * sigma[1] - 1
    return mu * sigmam12Sum + la / 2 * sigmaProdm1 * sigmaProdm1


@ti.func
def elasticity_gradient(sig: ti.template(), la, mu):
    sigma = ti.Matrix.zero(ti.get_runtime().default_fp, sig.n, 1)
    for i in ti.static(range(sig.n)):
        sigma[i] = sig[i, 0 if ti.static(sig.m == 1) else i]
    sigmaProdm1lambda = la * (sigma[0] * sigma[1] - 1)
    sigmaProd_noI = ti.Vector([sigma[1], sigma[0]])
    _2u = mu * 2
    return ti.Vector([_2u * (sigma[0] - 1) + sigmaProd_noI[0] * sigmaProdm1lambda,
                      _2u * (sigma[1] - 1) + sigmaProd_noI[1] * sigmaProdm1lambda])


@ti.func
def elasticity_hessian(sig: ti.template(), la, mu):
    sigma = ti.Matrix.zero(ti.get_runtime().default_fp, sig.n, 1)
    for i in ti.static(range(sig.n)):
        sigma[i] = sig[i, 0 if ti.static(sig.m == 1) else i]
    sigmaProd = sigma[0] * sigma[1]
    sigmaProd_noI = ti.Vector([sigma[1], sigma[0]])
    _2u = mu * 2
    return ti.Matrix([[_2u + la * sigmaProd_noI[0] * sigmaProd_noI[0],
                       la * ((sigmaProd - 1) + sigmaProd_noI[0] * sigmaProd_noI[1])],
                      [la * ((sigmaProd - 1) + sigmaProd_noI[0] * sigmaProd_noI[1]),
                       _2u + la * sigmaProd_noI[1] * sigmaProd_noI[1]]])


@ti.func
def elasticity_first_piola_kirchoff_stress(F, la, mu):
    J = F.determinant()
    JFinvT = cofactor(F)
    U, sig, V = svd(F)
    R = U @ V.transpose()
    return 2 * mu * (F - R) + la * (J - 1) * JFinvT


@ti.func
def elasticity_first_piola_kirchoff_stress_derivative(F, la, mu):
    U, sig, V = svd(F)
    sigma = ti.Vector([sig[0, 0], sig[1, 1]])
    dE_div_dsigma = fixed_corotated_gradient(sig, la, mu)
    d2E_div_dsigma2 = make_pd(fixed_corotated_hessian(sig, la, mu))

    leftCoef = mu - la / 2 * (sigma[0] * sigma[1] - 1)
    rightCoef = dE_div_dsigma[0] + dE_div_dsigma[1]
    sum_sigma = ti.max(sigma[0] + sigma[1], 0.000001)
    rightCoef /= (2 * sum_sigma)
    B = make_pd(ti.Matrix([[leftCoef + rightCoef, leftCoef - rightCoef], [leftCoef - rightCoef, leftCoef + rightCoef]]))

    M = ti.Matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    dPdF = ti.Matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    M[0, 0] = d2E_div_dsigma2[0, 0]
    M[0, 3] = d2E_div_dsigma2[0, 1]
    M[1, 1] = B[0, 0]
    M[1, 2] = B[0, 1]
    M[2, 1] = B[1, 0]
    M[2, 2] = B[1, 1]
    M[3, 0] = d2E_div_dsigma2[1, 0]
    M[3, 3] = d2E_div_dsigma2[1, 1]
    for j in ti.static(range(2)):
        for i in ti.static(range(2)):
            for s in ti.static(range(2)):
                for r in ti.static(range(2)):
                    ij = ti.static(j * 2 + i)
                    rs = ti.static(s * 2 + r)
                    dPdF[ij, rs] = M[0, 0] * U[i, 0] * V[j, 0] * U[r, 0] * V[s, 0] + M[0, 3] * U[i, 0] * V[j, 0] * U[r, 1] * V[s, 1] + M[1, 1] * U[i, 0] * V[j, 1] * U[r, 0] * V[s, 1] + M[1, 2] * U[i, 0] * V[j, 1] * U[r, 1] * V[s, 0] + M[2, 1] * U[i, 1] * V[j, 0] * U[r, 0] * V[s, 1] + M[2, 2] * U[i, 1] * V[j, 0] * U[r, 1] * V[s, 0] + M[3, 0] * U[i, 1] * V[j, 1] * U[r, 0] * V[s, 0] + M[3, 3] * U[i, 1] * V[j, 1] * U[r, 1] * V[s, 1]
    return dPdF

