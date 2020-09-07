import taichi as ti
from math_tools import *


@ti.func
def elasticity_energy(sig, la, mu):
    sigma = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
    sigma2Sum = sigma.norm_sqr()
    sigmaProd = sigma[0] * sigma[1] * sigma[2]
    log_sigmaProd = ti.log(sigmaProd)
    return mu / 2.0 * (sigma2Sum - 3) - (mu - la / 2.0 * log_sigmaProd) * log_sigmaProd


@ti.func
def elasticity_gradient(sig, la, mu):
    sigma = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
    sigmaProd = sigma[0] * sigma[1] * sigma[2]
    log_sigmaProd = ti.log(sigmaProd)
    inv0 = 1.0 / sigma[0]
    inv1 = 1.0 / sigma[1]
    inv2 = 1.0 / sigma[2]
    return ti.Vector([mu * (sigma[0] - inv0) + la * inv0 * log_sigmaProd,
                      mu * (sigma[1] - inv1) + la * inv1 * log_sigmaProd,
                      mu * (sigma[2] - inv2) + la * inv2 * log_sigmaProd])


@ti.func
def elasticity_hessian(sig, la, mu):
    sigma = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
    sigmaProd = sigma[0] * sigma[1] * sigma[2]
    log_sigmaProd = ti.log(sigmaProd)
    inv2_0 = 1.0 / sigma[0] / sigma[0]
    inv2_1 = 1.0 / sigma[1] / sigma[1]
    inv2_2 = 1.0 / sigma[2] / sigma[2]
    H00 = mu * (1.0 + inv2_0) - la * inv2_0 * (log_sigmaProd - 1.0)
    H11 = mu * (1.0 + inv2_1) - la * inv2_1 * (log_sigmaProd - 1.0)
    H22 = mu * (1.0 + inv2_2) - la * inv2_2 * (log_sigmaProd - 1.0)
    H01 = la / sigma[0] / sigma[1]
    H12 = la / sigma[1] / sigma[2]
    H02 = la / sigma[0] / sigma[2]
    return ti.Matrix([[H00, H01, H02],
                      [H01, H11, H12],
                      [H02, H12, H22]])


@ti.func
def elasticity_first_piola_kirchoff_stress(F, la, mu):
    J = F.determinant()
    logJ = ti.log(J)
    scale = la * logJ - mu
    tau = mu * F * F.transpose() + scale * ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    FinvT = F.inverse().transpose()
    return tau * FinvT


@ti.func
def elasticity_first_piola_kirchoff_stress_derivative(F, la, mu):
    U, sig, V = ti.svd(F)
    sigma = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
    sigmaProd = sigma[0] * sigma[1] * sigma[2]
    middle = mu - la * ti.log(sigmaProd)
    dE_div_dsigma = elasticity_gradient(sig, la, mu)
    d2E_div_dsigma2 = project_pd_3(elasticity_hessian(sig, la, mu))

    leftCoef = (mu + middle / sigma[0] / sigma[1]) / 2.0
    rightCoef = dE_div_dsigma[0] + dE_div_dsigma[1]
    sum_sigma = ti.max(sigma[0] + sigma[1], 0.000001)
    rightCoef /= (2 * sum_sigma)
    B0 = make_pd(ti.Matrix([[leftCoef + rightCoef, leftCoef - rightCoef], [leftCoef - rightCoef, leftCoef + rightCoef]]))

    leftCoef = (mu + middle / sigma[1] / sigma[2]) / 2.0
    rightCoef = dE_div_dsigma[1] + dE_div_dsigma[2]
    sum_sigma = ti.max(sigma[1] + sigma[2], 0.000001)
    rightCoef /= (2 * sum_sigma)
    B1 = make_pd(ti.Matrix([[leftCoef + rightCoef, leftCoef - rightCoef], [leftCoef - rightCoef, leftCoef + rightCoef]]))

    leftCoef = (mu + middle / sigma[2] / sigma[0]) / 2.0
    rightCoef = dE_div_dsigma[2] + dE_div_dsigma[0]
    sum_sigma = ti.max(sigma[2] + sigma[0], 0.000001)
    rightCoef /= (2 * sum_sigma)
    B2 = make_pd(ti.Matrix([[leftCoef + rightCoef, leftCoef - rightCoef], [leftCoef - rightCoef, leftCoef + rightCoef]]))

    Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    M = ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z])
    dPdF = ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z])
    M[0, 0] = d2E_div_dsigma2[0, 0]
    M[0, 4] = d2E_div_dsigma2[0, 1]
    M[0, 8] = d2E_div_dsigma2[0, 2]
    M[4, 0] = d2E_div_dsigma2[1, 0]
    M[4, 4] = d2E_div_dsigma2[1, 1]
    M[4, 8] = d2E_div_dsigma2[1, 2]
    M[8, 0] = d2E_div_dsigma2[2, 0]
    M[8, 4] = d2E_div_dsigma2[2, 1]
    M[8, 8] = d2E_div_dsigma2[2, 2]
    M[1, 1] = B0[0, 0]
    M[1, 3] = B0[0, 1]
    M[3, 1] = B0[1, 0]
    M[3, 3] = B0[1, 1]
    M[5, 5] = B1[0, 0]
    M[5, 7] = B1[0, 1]
    M[7, 5] = B1[1, 0]
    M[7, 7] = B1[1, 1]
    M[2, 2] = B2[1, 1]
    M[2, 6] = B2[1, 0]
    M[6, 2] = B2[0, 1]
    M[6, 6] = B2[0, 0]
    for j in ti.static(range(3)):
        for i in ti.static(range(3)):
            for s in ti.static(range(3)):
                for r in ti.static(range(3)):
                    ij = ti.static(j * 3 + i)
                    rs = ti.static(s * 3 + r)
                    dPdF[ij, rs] = M[0, 0] * U[i, 0] * V[j, 0] * U[r, 0] * V[s, 0] + M[0, 4] * U[i, 0] * V[j, 0] * U[r, 1] * V[s, 1] + M[0, 8] * U[i, 0] * V[j, 0] * U[r, 2] * V[s, 2] + M[4, 0] * U[i, 1] * V[j, 1] * U[r, 0] * V[s, 0] + M[4, 4] * U[i, 1] * V[j, 1] * U[r, 1] * V[s, 1] + M[4, 8] * U[i, 1] * V[j, 1] * U[r, 2] * V[s, 2] + M[8, 0] * U[i, 2] * V[j, 2] * U[r, 0] * V[s, 0] + M[8, 4] * U[i, 2] * V[j, 2] * U[r, 1] * V[s, 1] + M[8, 8] * U[i, 2] * V[j, 2] * U[r, 2] * V[s, 2] + M[1, 1] * U[i, 0] * V[j, 1] * U[r, 0] * V[s, 1] + M[1, 3] * U[i, 0] * V[j, 1] * U[r, 1] * V[s, 0] + M[3, 1] * U[i, 1] * V[j, 0] * U[r, 0] * V[s, 1] + M[3, 3] * U[i, 1] * V[j, 0] * U[r, 1] * V[s, 0] + M[5, 5] * U[i, 1] * V[j, 2] * U[r, 1] * V[s, 2] + M[5, 7] * U[i, 1] * V[j, 2] * U[r, 2] * V[s, 1] + M[7, 5] * U[i, 2] * V[j, 1] * U[r, 1] * V[s, 2] + M[7, 7] * U[i, 2] * V[j, 1] * U[r, 2] * V[s, 1] + M[2, 2] * U[i, 0] * V[j, 2] * U[r, 0] * V[s, 2] + M[2, 6] * U[i, 0] * V[j, 2] * U[r, 2] * V[s, 0] + M[6, 2] * U[i, 2] * V[j, 0] * U[r, 0] * V[s, 2] + M[6, 6] * U[i, 2] * V[j, 0] * U[r, 2] * V[s, 0]
    return dPdF

