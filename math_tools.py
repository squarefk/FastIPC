import taichi as ti
import ctypes
import numpy as np


def make_semi_positive_definite(mat, n):
    w, v = np.linalg.eig(mat.reshape((n, n)))
    return (v @ np.diag(np.maximum(w, 0)) @ v.transpose()).reshape((n * n))


@ti.func
def fill_vec(v, idx: ti.template(), t: ti.template()):
    vec = ti.Matrix.zero(ti.f32, 12)
    for i in ti.static(range(t)):
        vec[idx[i]] = v[i]
    return vec


@ti.func
def fill_mat(m, idx: ti.template(), t: ti.template()):
    mat = ti.Matrix.zero(ti.f32, 12, 12)
    for i in ti.static(range(t)):
        for j in ti.static(range(t)):
            mat[idx[i], idx[j]] = m[i, j]
    return mat


@ti.func
def make_pd(symMtr):
    a = symMtr[0, 0]
    b = (symMtr[0, 1] + symMtr[1, 0]) / 2.0
    d = symMtr[1, 1]
    b2 = b * b
    D = a * d - b2
    T_div_2 = (a + d) / 2.0
    sqrtTT4D = ti.sqrt(T_div_2 * T_div_2 - D)
    L2 = T_div_2 - sqrtTT4D
    if L2 < 0.0:
        L1 = T_div_2 + sqrtTT4D
        if L1 <= 0.0:
            symMtr = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        else:
            if b2 == 0.0:
                symMtr = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                symMtr[0, 0] = L1
            else:
                L1md = L1 - d
                L1md_div_L1 = L1md / L1
                symMtr[0, 0] = L1md_div_L1 * L1md
                symMtr[0, 1] = b * L1md_div_L1
                symMtr[1, 0] = b * L1md_div_L1
                symMtr[1, 1] = b2 / L1
    return symMtr


so = ctypes.CDLL("./wrapper/a.so")


@ti.func
def singular_value_decomposition(F):
    F00, F01, F10, F11 = F(0, 0), F(0, 1), F(1, 0), F(1, 1)
    U00, U01, U10, U11 = 0.0, 0.0, 0.0, 0.0
    s00, s01, s10, s11 = 0.0, 0.0, 0.0, 0.0
    V00, V01, V10, V11 = 0.0, 0.0, 0.0, 0.0
    ti.external_func_call(func=so.svd,
                          args=(F00, F01, F10, F11),
                          outputs=(U00, U01, U10, U11, s00, s01, s10, s11, V00, V01, V10, V11))
    return ti.Matrix([[U00, U01], [U10, U11]]), ti.Matrix([[s00, s01], [s10, s11]]), ti.Matrix([[V00, V01], [V10, V11]])

@ti.func
def project_pd3(F):
    F00, F01, F02 = F(0, 0), F(0, 1), F(0, 2)
    F10, F11, F12 = F(1, 0), F(1, 1), F(1, 2)
    F20, F21, F22 = F(2, 0), F(2, 1), F(2, 2)
    PF00, PF01, PF02 = 0.0, 0.0, 0.0
    PF10, PF11, PF12 = 0.0, 0.0, 0.0
    PF20, PF21, PF22 = 0.0, 0.0, 0.0
    ti.external_func_call(func=so.project_pd3,
                          args=(F00, F01, F02,
                                F10, F11, F12,
                                F20, F21, F22),
                          outputs=(PF00, PF01, PF02,
                                   PF10, PF11, PF12,
                                   PF20, PF21, PF22))
    return ti.Matrix([[PF00, PF01, PF02],
                      [PF10, PF11, PF12],
                      [PF20, PF21, PF22]])


@ti.func
def project_pd(F, diagonal):
    F00, F01, F02, F03, F04, F05 = F(0, 0), F(0, 1), F(0, 2), F(0, 3), F(0, 4), F(0, 5)
    F10, F11, F12, F13, F14, F15 = F(1, 0), F(1, 1), F(1, 2), F(1, 3), F(1, 4), F(1, 5)
    F20, F21, F22, F23, F24, F25 = F(2, 0), F(2, 1), F(2, 2), F(2, 3), F(2, 4), F(2, 5)
    F30, F31, F32, F33, F34, F35 = F(3, 0), F(3, 1), F(3, 2), F(3, 3), F(3, 4), F(3, 5)
    F40, F41, F42, F43, F44, F45 = F(4, 0), F(4, 1), F(4, 2), F(4, 3), F(4, 4), F(4, 5)
    F50, F51, F52, F53, F54, F55 = F(5, 0), F(5, 1), F(5, 2), F(5, 3), F(5, 4), F(5, 5)
    PF00, PF01, PF02, PF03, PF04, PF05 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    PF10, PF11, PF12, PF13, PF14, PF15 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    PF20, PF21, PF22, PF23, PF24, PF25 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    PF30, PF31, PF32, PF33, PF34, PF35 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    PF40, PF41, PF42, PF43, PF44, PF45 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    PF50, PF51, PF52, PF53, PF54, PF55 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ti.external_func_call(func=so.project_pd,
                          args=(F00, F01, F02, F03, F04, F05,
                                F10, F11, F12, F13, F14, F15,
                                F20, F21, F22, F23, F24, F25,
                                F30, F31, F32, F33, F34, F35,
                                F40, F41, F42, F43, F44, F45,
                                F50, F51, F52, F53, F54, F55, diagonal),
                          outputs=(PF00, PF01, PF02, PF03, PF04, PF05,
                                   PF10, PF11, PF12, PF13, PF14, PF15,
                                   PF20, PF21, PF22, PF23, PF24, PF25,
                                   PF30, PF31, PF32, PF33, PF34, PF35,
                                   PF40, PF41, PF42, PF43, PF44, PF45,
                                   PF50, PF51, PF52, PF53, PF54, PF55))
    return ti.Matrix([[PF00, PF01, PF02, PF03, PF04, PF05],
                      [PF10, PF11, PF12, PF13, PF14, PF15],
                      [PF20, PF21, PF22, PF23, PF24, PF25],
                      [PF30, PF31, PF32, PF33, PF34, PF35],
                      [PF40, PF41, PF42, PF43, PF44, PF45],
                      [PF50, PF51, PF52, PF53, PF54, PF55]])



@ti.func
def project_pd64(F, diagonal):
    F00, F01, F02, F03, F04, F05 = F(0, 0), F(0, 1), F(0, 2), F(0, 3), F(0, 4), F(0, 5)
    F10, F11, F12, F13, F14, F15 = F(1, 0), F(1, 1), F(1, 2), F(1, 3), F(1, 4), F(1, 5)
    F20, F21, F22, F23, F24, F25 = F(2, 0), F(2, 1), F(2, 2), F(2, 3), F(2, 4), F(2, 5)
    F30, F31, F32, F33, F34, F35 = F(3, 0), F(3, 1), F(3, 2), F(3, 3), F(3, 4), F(3, 5)
    F40, F41, F42, F43, F44, F45 = F(4, 0), F(4, 1), F(4, 2), F(4, 3), F(4, 4), F(4, 5)
    F50, F51, F52, F53, F54, F55 = F(5, 0), F(5, 1), F(5, 2), F(5, 3), F(5, 4), F(5, 5)
    PF00, PF01, PF02, PF03 = 0.0, 0.0, 0.0, 0.0
    PF10, PF11, PF12, PF13 = 0.0, 0.0, 0.0, 0.0
    PF20, PF21, PF22, PF23 = 0.0, 0.0, 0.0, 0.0
    PF30, PF31, PF32, PF33 = 0.0, 0.0, 0.0, 0.0
    ti.external_func_call(func=so.project_pd64,
                          args=(F00, F01, F02, F03, F04, F05,
                                F10, F11, F12, F13, F14, F15,
                                F20, F21, F22, F23, F24, F25,
                                F30, F31, F32, F33, F34, F35,
                                F40, F41, F42, F43, F44, F45,
                                F50, F51, F52, F53, F54, F55, diagonal),
                          outputs=(PF00, PF01, PF02, PF03,
                                   PF10, PF11, PF12, PF13,
                                   PF20, PF21, PF22, PF23,
                                   PF30, PF31, PF32, PF33))
    return ti.Matrix([[PF00, PF01, PF02, PF03],
                      [PF10, PF11, PF12, PF13],
                      [PF20, PF21, PF22, PF23],
                      [PF30, PF31, PF32, PF33]])


@ti.func
def point_triangle_ccd(p, t0, t1, t2, dp, dt0, dt1, dt2, eta, dist2):
    toc = 0.0
    ti.external_func_call(func=so.point_triangle_ccd,
                          args=(p[0], p[1], p[2],
                                t0[0], t0[1], t0[2],
                                t1[0], t1[1], t1[2],
                                t2[0], t2[1], t2[2],
                                dp[0], dp[1], dp[2],
                                dt0[0], dt0[1], dt0[2],
                                dt1[0], dt1[1], dt1[2],
                                dt2[0], dt2[1], dt2[2], eta, dist2),
                          outputs=(toc,))
    return toc


@ti.func
def edge_edge_ccd(a0, a1, b0, b1, da0, da1, db0, db1, eta, dist2):
    toc = 0.0
    ti.external_func_call(func=so.edge_edge_ccd,
                          args=(a0[0], a0[1], a0[2],
                                a1[0], a1[1], a1[2],
                                b0[0], b0[1], b0[2],
                                b1[0], b1[1], b1[2],
                                da0[0], da0[1], da0[2],
                                da1[0], da1[1], da1[2],
                                db0[0], db0[1], db0[2],
                                db1[0], db1[1], db1[2], eta, dist2),
                          outputs=(toc,))
    return toc