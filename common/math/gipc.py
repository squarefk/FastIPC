from common.math.ipc import *


@ti.func
def GPE_energy(p, e0, e1, dHat2, kappa):
    e = e1 - e0
    ratio = e.dot(p - e0) / e.norm_sqr()
    E = 0.
    if ratio < 0:
        E = PP_energy(p, e0, dHat2, kappa)
    elif ratio > 1:
        E = PP_energy(p, e1, dHat2, kappa)
    else:
        E = PE_energy(p, e0, e1, dHat2, kappa)
    return E
@ti.func
def GPE_gradient(p, e0, e1, dHat2, kappa):
    e = e1 - e0
    ratio = e.dot(p - e0) / e.norm_sqr()
    g = ti.Matrix.zero(ti.get_runtime().default_fp, 6)
    if ratio < 0:
        g = fill_vec(PP_gradient(p, e0, dHat2, kappa), [0, 1, 2, 3], 6)
    elif ratio > 1:
        g = fill_vec(PP_gradient(p, e1, dHat2, kappa), [0, 1, 4, 5], 6)
    else:
        g = PE_gradient(p, e0, e1, dHat2, kappa)
    return g
@ti.func
def GPE_hessian(p, e0, e1, dHat2, kappa):
    e = e1 - e0
    ratio = e.dot(p - e0) / e.norm_sqr()
    H = ti.Matrix.zero(ti.get_runtime().default_fp, 6, 6)
    if ratio < 0:
        H = fill_mat(PP_hessian(p, e0, dHat2, kappa), [0, 1, 2, 3], 6)
    elif ratio > 1:
        H = fill_mat(PP_hessian(p, e1, dHat2, kappa), [0, 1, 4, 5], 6)
    else:
        H = PE_hessian(p, e0, e1, dHat2, kappa)
    return H


@ti.func
def GPT_energy(p, t0, t1, t2, dHat2, kappa):
    case = PT_type(p, t0, t1, t2)
    E = 0.
    if case == 0:
        E = PP_energy(p, t0, dHat2, kappa)
    elif case == 1:
        E = PP_energy(p, t1, dHat2, kappa)
    elif case == 2:
        E = PP_energy(p, t2, dHat2, kappa)
    elif case == 3:
        E = PE_energy(p, t0, t1, dHat2, kappa)
    elif case == 4:
        E = PE_energy(p, t1, t2, dHat2, kappa)
    elif case == 5:
        E = PE_energy(p, t2, t0, dHat2, kappa)
    elif case == 6:
        E = PT_energy(p, t0, t1, t2, dHat2, kappa)
    return E
@ti.func
def GPT_gradient(p, t0, t1, t2, dHat2, kappa):
    case = PT_type(p, t0, t1, t2)
    g = ti.Matrix.zero(ti.get_runtime().default_fp, 12)
    if case == 0:
        g = fill_vec(PP_gradient(p, t0, dHat2, kappa), [0, 1, 2, 3, 4, 5], 12)
    elif case == 1:
        g = fill_vec(PP_gradient(p, t1, dHat2, kappa), [0, 1, 2, 6, 7, 8], 12)
    elif case == 2:
        g = fill_vec(PP_gradient(p, t2, dHat2, kappa), [0, 1, 2, 9, 10, 11], 12)
    elif case == 3:
        g = fill_vec(PE_gradient(p, t0, t1, dHat2, kappa), [0, 1, 2, 3, 4, 5, 6, 7, 8], 12)
    elif case == 4:
        g = fill_vec(PE_gradient(p, t1, t2, dHat2, kappa), [0, 1, 2, 6, 7, 8, 9, 10, 11], 12)
    elif case == 5:
        g = fill_vec(PE_gradient(p, t2, t0, dHat2, kappa), [0, 1, 2, 9, 10, 11, 3, 4, 5], 12)
    elif case == 6:
        g = PT_gradient(p, t0, t1, t2, dHat2, kappa)
    return g
@ti.func
def GPT_hessian(p, t0, t1, t2, dHat2, kappa):
    case = PT_type(p, t0, t1, t2)
    H = ti.Matrix.zero(ti.get_runtime().default_fp, 12, 12)
    if case == 0:
        H = fill_mat(PP_hessian(p, t0, dHat2, kappa), [0, 1, 2, 3, 4, 5], 12)
    elif case == 1:
        H = fill_mat(PP_hessian(p, t1, dHat2, kappa), [0, 1, 2, 6, 7, 8], 12)
    elif case == 2:
        H = fill_mat(PP_hessian(p, t2, dHat2, kappa), [0, 1, 2, 9, 10, 11], 12)
    elif case == 3:
        H = fill_mat(PE_hessian(p, t0, t1, dHat2, kappa), [0, 1, 2, 3, 4, 5, 6, 7, 8], 12)
    elif case == 4:
        H = fill_mat(PE_hessian(p, t1, t2, dHat2, kappa), [0, 1, 2, 6, 7, 8, 9, 10, 11], 12)
    elif case == 5:
        H = fill_mat(PE_hessian(p, t2, t0, dHat2, kappa), [0, 1, 2, 9, 10, 11, 3, 4, 5], 12)
    elif case == 6:
        H = PT_hessian(p, t0, t1, t2, dHat2, kappa)
    return H


@ti.func
def GEE_energy(a0, a1, b0, b1, dHat2, kappa):
    case = EE_type(a0, a1, b0, b1)
    E = 0.
    if case == 0:
        E = PP_energy(a0, b0, dHat2, kappa)
    elif case == 1:
        E = PP_energy(a0, b1, dHat2, kappa)
    elif case == 2:
        E = PE_energy(a0, b0, b1, dHat2, kappa)
    elif case == 3:
        E = PP_energy(a1, b0, dHat2, kappa)
    elif case == 4:
        E = PP_energy(a1, b1, dHat2, kappa)
    elif case == 5:
        E = PE_energy(a1, b0, b1, dHat2, kappa)
    elif case == 6:
        E = PE_energy(b0, a0, a1, dHat2, kappa)
    elif case == 7:
        E = PE_energy(b1, a0, a1, dHat2, kappa)
    elif case == 8:
        E = EE_energy(a0, a1, b0, b1, dHat2, kappa)
    return E
@ti.func
def GEE_gradient(a0, a1, b0, b1, dHat2, kappa):
    case = EE_type(a0, a1, b0, b1)
    g = ti.Matrix.zero(ti.get_runtime().default_fp, 12)
    if case == 0:
        g = fill_vec(PP_gradient(a0, b0, dHat2, kappa), [0, 1, 2, 6, 7, 8], 12)
    elif case == 1:
        g = fill_vec(PP_gradient(a0, b1, dHat2, kappa), [0, 1, 2, 9, 10, 11], 12)
    elif case == 2:
        g = fill_vec(PE_gradient(a0, b0, b1, dHat2, kappa), [0, 1, 2, 6, 7, 8, 9, 10, 11], 12)
    elif case == 3:
        g = fill_vec(PP_gradient(a1, b0, dHat2, kappa), [3, 4, 5, 6, 7, 8], 12)
    elif case == 4:
        g = fill_vec(PP_gradient(a1, b1, dHat2, kappa), [3, 4, 5, 9, 10, 11], 12)
    elif case == 5:
        g = fill_vec(PE_gradient(a1, b0, b1, dHat2, kappa), [3, 4, 5, 6, 7, 8, 9, 10, 11], 12)
    elif case == 6:
        g = fill_vec(PE_gradient(b0, a0, a1, dHat2, kappa), [6, 7, 8, 0, 1, 2, 3, 4, 5], 12)
    elif case == 7:
        g = fill_vec(PE_gradient(b1, a0, a1, dHat2, kappa), [9, 10, 11, 0, 1, 2, 3, 4, 5], 12)
    elif case == 8:
        g = EE_gradient(a0, a1, b0, b1, dHat2, kappa)
    return g
@ti.func
def GEE_hessian(a0, a1, b0, b1, dHat2, kappa):
    case = EE_type(a0, a1, b0, b1)
    H = ti.Matrix.zero(ti.get_runtime().default_fp, 12, 12)
    if case == 0:
        H = fill_mat(PP_hessian(a0, b0, dHat2, kappa), [0, 1, 2, 6, 7, 8], 12)
    elif case == 1:
        H = fill_mat(PP_hessian(a0, b1, dHat2, kappa), [0, 1, 2, 9, 10, 11], 12)
    elif case == 2:
        H = fill_mat(PE_hessian(a0, b0, b1, dHat2, kappa), [0, 1, 2, 6, 7, 8, 9, 10, 11], 12)
    elif case == 3:
        H = fill_mat(PP_hessian(a1, b0, dHat2, kappa), [3, 4, 5, 6, 7, 8], 12)
    elif case == 4:
        H = fill_mat(PP_hessian(a1, b1, dHat2, kappa), [3, 4, 5, 9, 10, 11], 12)
    elif case == 5:
        H = fill_mat(PE_hessian(a1, b0, b1, dHat2, kappa), [3, 4, 5, 6, 7, 8, 9, 10, 11], 12)
    elif case == 6:
        H = fill_mat(PE_hessian(b0, a0, a1, dHat2, kappa), [6, 7, 8, 0, 1, 2, 3, 4, 5], 12)
    elif case == 7:
        H = fill_mat(PE_hessian(b1, a0, a1, dHat2, kappa), [9, 10, 11, 0, 1, 2, 3, 4, 5], 12)
    elif case == 8:
        H = EE_hessian(a0, a1, b0, b1, dHat2, kappa)
    return H



@ti.func
def GEEM_energy(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    case = EE_type(a0, a1, b0, b1)
    E = 0.
    if case == 0:
        E = PPM_energy(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa)
    elif case == 1:
        E = PPM_energy(a0, a1, b1, b0, _a0, _a1, _b1, _b0, dHat2, kappa)
    elif case == 2:
        E = PEM_energy(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa)
    elif case == 3:
        E = PPM_energy(a1, a0, b0, b1, _a1, _a0, _b0, _b1, dHat2, kappa)
    elif case == 4:
        E = PPM_energy(a1, a0, b1, b0, _a1, _a0, _b1, _b0, dHat2, kappa)
    elif case == 5:
        E = PEM_energy(a1, a0, b0, b1, _a1, _a0, _b0, _b1, dHat2, kappa)
    elif case == 6:
        E = PEM_energy(b0, b1, a0, a1, _b0, _b1, _a0, _a1, dHat2, kappa)
    elif case == 7:
        E = PEM_energy(b1, b0, a0, a1, _b1, _b0, _a0, _a1, dHat2, kappa)
    elif case == 8:
        E = EEM_energy(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa)
    return E
@ti.func
def GEEM_gradient(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    case = EE_type(a0, a1, b0, b1)
    g = ti.Matrix.zero(ti.get_runtime().default_fp, 12)
    if case == 0:
        g = PPM_gradient(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa)
    elif case == 1:
        g = PPM_gradient(a0, a1, b1, b0, _a0, _a1, _b1, _b0, dHat2, kappa)
    elif case == 2:
        g = PEM_gradient(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa)
    elif case == 3:
        g = PPM_gradient(a1, a0, b0, b1, _a1, _a0, _b0, _b1, dHat2, kappa)
    elif case == 4:
        g = PPM_gradient(a1, a0, b1, b0, _a1, _a0, _b1, _b0, dHat2, kappa)
    elif case == 5:
        g = PEM_gradient(a1, a0, b0, b1, _a1, _a0, _b0, _b1, dHat2, kappa)
    elif case == 6:
        g = PEM_gradient(b0, b1, a0, a1, _b0, _b1, _a0, _a1, dHat2, kappa)
    elif case == 7:
        g = PEM_gradient(b1, b0, a0, a1, _b1, _b0, _a0, _a1, dHat2, kappa)
    elif case == 8:
        g = EEM_gradient(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa)
    return g
@ti.func
def GEEM_hessian(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    case = EE_type(a0, a1, b0, b1)
    H = ti.Matrix.zero(ti.get_runtime().default_fp, 12, 12)
    if case == 0:
        H = PPM_hessian(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa)
    elif case == 1:
        H = PPM_hessian(a0, a1, b1, b0, _a0, _a1, _b1, _b0, dHat2, kappa)
    elif case == 2:
        H = PEM_hessian(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa)
    elif case == 3:
        H = PPM_hessian(a1, a0, b0, b1, _a1, _a0, _b0, _b1, dHat2, kappa)
    elif case == 4:
        H = PPM_hessian(a1, a0, b1, b0, _a1, _a0, _b1, _b0, dHat2, kappa)
    elif case == 5:
        H = PEM_hessian(a1, a0, b0, b1, _a1, _a0, _b0, _b1, dHat2, kappa)
    elif case == 6:
        H = PEM_hessian(b0, b1, a0, a1, _b0, _b1, _a0, _a1, dHat2, kappa)
    elif case == 7:
        H = PEM_hessian(b1, b0, a0, a1, _b1, _b0, _a0, _a1, dHat2, kappa)
    elif case == 8:
        H = EEM_hessian(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa)
    return H
