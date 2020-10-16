import taichi as ti
from distance import *
from math_tools import *


@ti.func
def PP_energy(p0, p1, dHat2, kappa):
    if ti.static(p0.n == 2):
        dist2 = PP_2D_E(p0, p1)
        if dist2 < 1e-12:
            print("ERROR PP", dist2)
        return barrier_E(dist2, dHat2, kappa)
    else:
        dist2 = PP_3D_E(p0, p1)
        if dist2 < 1e-12:
            print("ERROR PP", dist2)
        return barrier_E(dist2, dHat2, kappa)
@ti.func
def PP_gradient(p0, p1, dHat2, kappa):
    if ti.static(p0.n == 2):
        dist2 = PP_2D_E(p0, p1)
        dist2g = PP_2D_g(p0, p1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        return ti.Vector([g[2], g[3]])
    else:
        dist2 = PP_3D_E(p0, p1)
        dist2g = PP_3D_g(p0, p1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        return ti.Vector([g[3], g[4], g[5]])
@ti.func
def PP_hessian(p0, p1, dHat2, kappa):
    if ti.static(p0.n == 2):
        dist2 = PP_2D_E(p0, p1)
        dist2g = PP_2D_g(p0, p1)
        bg = barrier_g(dist2, dHat2, kappa)
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_2D_H(p0, p1)
        eH = ti.Matrix([[H[2, 2], H[2, 3]], [H[3, 2], H[3, 3]]])
        return project_pd(eH)
    else:
        dist2 = PP_3D_E(p0, p1)
        dist2g = PP_3D_g(p0, p1)
        bg = barrier_g(dist2, dHat2, kappa)
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_3D_H(p0, p1)
        eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5]], [H[4, 3], H[4, 4], H[4, 5]], [H[5, 3], H[5, 4], H[5, 5]]])
        return project_pd(eH)


@ti.func
def PE_energy(p, e0, e1, dHat2, kappa):
    if ti.static(p.n == 2):
        dist2 = PE_2D_E(p, e0, e1)
        if dist2 < 1e-12:
            print("ERROR PE", dist2)
        return barrier_E(dist2, dHat2, kappa)
    else:
        dist2 = PE_3D_E(p, e0, e1)
        if dist2 < 1e-12:
            print("ERROR PE", dist2)
        return barrier_E(dist2, dHat2, kappa)
@ti.func
def PE_gradient(p, e0, e1, dHat2, kappa):
    if ti.static(p.n == 2):
        dist2 = PE_2D_E(p, e0, e1)
        dist2g = PE_2D_g(p, e0, e1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        return ti.Vector([g[2], g[3], g[4], g[5]])
    else:
        dist2 = PE_3D_E(p, e0, e1)
        dist2g = PE_3D_g(p, e0, e1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        return ti.Vector([g[3], g[4], g[5], g[6], g[7], g[8]])
@ti.func
def PE_hessian(p, e0, e1, dHat2, kappa):
    if ti.static(p.n == 2):
        dist2 = PE_2D_E(p, e0, e1)
        dist2g = PE_2D_g(p, e0, e1)
        bg = barrier_g(dist2, dHat2, kappa)
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_2D_H(p, e0, e1)
        eH = ti.Matrix([[H[2, 2], H[2, 3], H[2, 4], H[2, 5]], [H[3, 2], H[3, 3], H[3, 4], H[3, 5]], [H[4, 2], H[4, 3], H[4, 4], H[4, 5]], [H[5, 2], H[5, 3], H[5, 4], H[5, 5]]])
        return project_pd(eH)
    else:
        dist2 = PE_3D_E(p, e0, e1)
        dist2g = PE_3D_g(p, e0, e1)
        bg = barrier_g(dist2, dHat2, kappa)
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_3D_H(p, e0, e1)
        eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5], H[3, 6], H[3, 7], H[3, 8]], [H[4, 3], H[4, 4], H[4, 5], H[4, 6], H[4, 7], H[4, 8]], [H[5, 3], H[5, 4], H[5, 5], H[5, 6], H[5, 7], H[5, 8]], [H[6, 3], H[6, 4], H[6, 5], H[6, 6], H[6, 7], H[6, 8]], [H[7, 3], H[7, 4], H[7, 5], H[7, 6], H[7, 7], H[7, 8]], [H[8, 3], H[8, 4], H[8, 5], H[8, 6], H[8, 7], H[8, 8]]])
        return project_pd(eH)


@ti.func
def PT_energy(p, t0, t1, t2, dHat2, kappa):
    dist2 = PT_3D_E(p, t0, t1, t2)
    if dist2 < 1e-12:
        print("ERROR PT", dist2)
    return barrier_E(dist2, dHat2, kappa)
@ti.func
def PT_gradient(p, t0, t1, t2, dHat2, kappa):
    dist2 = PT_3D_E(p, t0, t1, t2)
    dist2g = PT_3D_g(p, t0, t1, t2)
    bg = barrier_g(dist2, dHat2, kappa)
    g = bg * dist2g
    return ti.Vector([g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11]])
@ti.func
def PT_hessian(p, t0, t1, t2, dHat2, kappa):
    dist2 = PT_3D_E(p, t0, t1, t2)
    dist2g = PT_3D_g(p, t0, t1, t2)
    bg = barrier_g(dist2, dHat2, kappa)
    H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PT_3D_H(p, t0, t1, t2)
    eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5], H[3, 6], H[3, 7], H[3, 8], H[3, 9], H[3, 10], H[3, 11]], [H[4, 3], H[4, 4], H[4, 5], H[4, 6], H[4, 7], H[4, 8], H[4, 9], H[4, 10], H[4, 11]], [H[5, 3], H[5, 4], H[5, 5], H[5, 6], H[5, 7], H[5, 8], H[5, 9], H[5, 10], H[5, 11]], [H[6, 3], H[6, 4], H[6, 5], H[6, 6], H[6, 7], H[6, 8], H[6, 9], H[6, 10], H[6, 11]], [H[7, 3], H[7, 4], H[7, 5], H[7, 6], H[7, 7], H[7, 8], H[7, 9], H[7, 10], H[7, 11]], [H[8, 3], H[8, 4], H[8, 5], H[8, 6], H[8, 7], H[8, 8], H[8, 9], H[8, 10], H[8, 11]], [H[9, 3], H[9, 4], H[9, 5], H[9, 6], H[9, 7], H[9, 8], H[9, 9], H[9, 10], H[9, 11]], [H[10, 3], H[10, 4], H[10, 5], H[10, 6], H[10, 7], H[10, 8], H[10, 9], H[10, 10], H[10, 11]], [H[11, 3], H[11, 4], H[11, 5], H[11, 6], H[11, 7], H[11, 8], H[11, 9], H[11, 10], H[11, 11]]])
    return project_pd(eH)


@ti.func
def EE_energy(a0, a1, b0, b1, dHat2, kappa):
    dist2 = EE_3D_E(a0, a1, b0, b1)
    if dist2 < 1e-12:
        print("ERROR EE", dist2)
    return barrier_E(dist2, dHat2, kappa)
@ti.func
def EE_gradient(a0, a1, b0, b1, dHat2, kappa):
    dist2 = EE_3D_E(a0, a1, b0, b1)
    dist2g = EE_3D_g(a0, a1, b0, b1)
    bg = barrier_g(dist2, dHat2, kappa)
    g = bg * dist2g
    return ti.Vector([g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11]])
@ti.func
def EE_hessian(a0, a1, b0, b1, dHat2, kappa):
    dist2 = EE_3D_E(a0, a1, b0, b1)
    dist2g = EE_3D_g(a0, a1, b0, b1)
    bg = barrier_g(dist2, dHat2, kappa)
    H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * EE_3D_H(a0, a1, b0, b1)
    eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5], H[3, 6], H[3, 7], H[3, 8], H[3, 9], H[3, 10], H[3, 11]], [H[4, 3], H[4, 4], H[4, 5], H[4, 6], H[4, 7], H[4, 8], H[4, 9], H[4, 10], H[4, 11]], [H[5, 3], H[5, 4], H[5, 5], H[5, 6], H[5, 7], H[5, 8], H[5, 9], H[5, 10], H[5, 11]], [H[6, 3], H[6, 4], H[6, 5], H[6, 6], H[6, 7], H[6, 8], H[6, 9], H[6, 10], H[6, 11]], [H[7, 3], H[7, 4], H[7, 5], H[7, 6], H[7, 7], H[7, 8], H[7, 9], H[7, 10], H[7, 11]], [H[8, 3], H[8, 4], H[8, 5], H[8, 6], H[8, 7], H[8, 8], H[8, 9], H[8, 10], H[8, 11]], [H[9, 3], H[9, 4], H[9, 5], H[9, 6], H[9, 7], H[9, 8], H[9, 9], H[9, 10], H[9, 11]], [H[10, 3], H[10, 4], H[10, 5], H[10, 6], H[10, 7], H[10, 8], H[10, 9], H[10, 10], H[10, 11]], [H[11, 3], H[11, 4], H[11, 5], H[11, 6], H[11, 7], H[11, 8], H[11, 9], H[11, 10], H[11, 11]]])
    return project_pd(eH)


@ti.func
def EEM_energy(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = EE_3D_E(a0, a1, b0, b1)
    if dist2 < 1e-12:
        print("ERROR EEM", dist2)
    return barrier_E(dist2, dHat2, kappa) * M_E(a0, a1, b0, b1, eps_x)
@ti.func
def EEM_gradient(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = EE_3D_E(a0, a1, b0, b1)
    dist2g = EE_3D_g(a0, a1, b0, b1)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    lg = bg * dist2g
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    g = lg * M + b * Mg
    return ti.Vector([g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11]])
@ti.func
def EEM_hessian(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = EE_3D_E(a0, a1, b0, b1)
    dist2g = EE_3D_g(a0, a1, b0, b1)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    lg = bg * dist2g
    lH = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * EE_3D_H(a0, a1, b0, b1)
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    H = lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
    eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5], H[3, 6], H[3, 7], H[3, 8], H[3, 9], H[3, 10], H[3, 11]], [H[4, 3], H[4, 4], H[4, 5], H[4, 6], H[4, 7], H[4, 8], H[4, 9], H[4, 10], H[4, 11]], [H[5, 3], H[5, 4], H[5, 5], H[5, 6], H[5, 7], H[5, 8], H[5, 9], H[5, 10], H[5, 11]], [H[6, 3], H[6, 4], H[6, 5], H[6, 6], H[6, 7], H[6, 8], H[6, 9], H[6, 10], H[6, 11]], [H[7, 3], H[7, 4], H[7, 5], H[7, 6], H[7, 7], H[7, 8], H[7, 9], H[7, 10], H[7, 11]], [H[8, 3], H[8, 4], H[8, 5], H[8, 6], H[8, 7], H[8, 8], H[8, 9], H[8, 10], H[8, 11]], [H[9, 3], H[9, 4], H[9, 5], H[9, 6], H[9, 7], H[9, 8], H[9, 9], H[9, 10], H[9, 11]], [H[10, 3], H[10, 4], H[10, 5], H[10, 6], H[10, 7], H[10, 8], H[10, 9], H[10, 10], H[10, 11]], [H[11, 3], H[11, 4], H[11, 5], H[11, 6], H[11, 7], H[11, 8], H[11, 9], H[11, 10], H[11, 11]]])
    return project_pd(eH)

@ti.func
def PPM_energy(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PP_3D_E(a0, b0)
    if dist2 < 1e-12:
        print("ERROR EPPM", dist2)
    return barrier_E(dist2, dHat2, kappa) * M_E(a0, a1, b0, b1, eps_x)
@ti.func
def PPM_gradient(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PP_3D_E(a0, b0)
    dist2g = PP_3D_g(a0, b0)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    lg = fill_vec(bg * dist2g, [0, 1, 2, 6, 7, 8])
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    g = lg * M + b * Mg
    return ti.Vector([g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11]])
@ti.func
def PPM_hessian(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PP_3D_E(a0, b0)
    dist2g = PP_3D_g(a0, b0)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    lg = fill_vec(bg * dist2g, [0, 1, 2, 6, 7, 8])
    lH = fill_mat(barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_3D_H(a0, b0), [0, 1, 2, 6, 7, 8])
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    H = lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
    eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5], H[3, 6], H[3, 7], H[3, 8], H[3, 9], H[3, 10], H[3, 11]], [H[4, 3], H[4, 4], H[4, 5], H[4, 6], H[4, 7], H[4, 8], H[4, 9], H[4, 10], H[4, 11]], [H[5, 3], H[5, 4], H[5, 5], H[5, 6], H[5, 7], H[5, 8], H[5, 9], H[5, 10], H[5, 11]], [H[6, 3], H[6, 4], H[6, 5], H[6, 6], H[6, 7], H[6, 8], H[6, 9], H[6, 10], H[6, 11]], [H[7, 3], H[7, 4], H[7, 5], H[7, 6], H[7, 7], H[7, 8], H[7, 9], H[7, 10], H[7, 11]], [H[8, 3], H[8, 4], H[8, 5], H[8, 6], H[8, 7], H[8, 8], H[8, 9], H[8, 10], H[8, 11]], [H[9, 3], H[9, 4], H[9, 5], H[9, 6], H[9, 7], H[9, 8], H[9, 9], H[9, 10], H[9, 11]], [H[10, 3], H[10, 4], H[10, 5], H[10, 6], H[10, 7], H[10, 8], H[10, 9], H[10, 10], H[10, 11]], [H[11, 3], H[11, 4], H[11, 5], H[11, 6], H[11, 7], H[11, 8], H[11, 9], H[11, 10], H[11, 11]]])
    return project_pd(eH)


@ti.func
def PEM_energy(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PE_3D_E(a0, b0, b1)
    if dist2 < 1e-12:
        print("ERROR PEM", dist2)
    return barrier_E(dist2, dHat2, kappa) * M_E(a0, a1, b0, b1, eps_x)
@ti.func
def PEM_gradient(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PE_3D_E(a0, b0, b1)
    dist2g = PE_3D_g(a0, b0, b1)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    lg = fill_vec(bg * dist2g, [0, 1, 2, 6, 7, 8, 9, 10, 11])
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    g = lg * M + b * Mg
    return ti.Vector([g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11]])
@ti.func
def PEM_hessian(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PE_3D_E(a0, b0, b1)
    dist2g = PE_3D_g(a0, b0, b1)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    lg = fill_vec(bg * dist2g, [0, 1, 2, 6, 7, 8, 9, 10, 11])
    lH = fill_mat(barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_3D_H(a0, b0, b1), [0, 1, 2, 6, 7, 8, 9, 10, 11])
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    H = lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
    eH = ti.Matrix([[H[3, 3], H[3, 4], H[3, 5], H[3, 6], H[3, 7], H[3, 8], H[3, 9], H[3, 10], H[3, 11]], [H[4, 3], H[4, 4], H[4, 5], H[4, 6], H[4, 7], H[4, 8], H[4, 9], H[4, 10], H[4, 11]], [H[5, 3], H[5, 4], H[5, 5], H[5, 6], H[5, 7], H[5, 8], H[5, 9], H[5, 10], H[5, 11]], [H[6, 3], H[6, 4], H[6, 5], H[6, 6], H[6, 7], H[6, 8], H[6, 9], H[6, 10], H[6, 11]], [H[7, 3], H[7, 4], H[7, 5], H[7, 6], H[7, 7], H[7, 8], H[7, 9], H[7, 10], H[7, 11]], [H[8, 3], H[8, 4], H[8, 5], H[8, 6], H[8, 7], H[8, 8], H[8, 9], H[8, 10], H[8, 11]], [H[9, 3], H[9, 4], H[9, 5], H[9, 6], H[9, 7], H[9, 8], H[9, 9], H[9, 10], H[9, 11]], [H[10, 3], H[10, 4], H[10, 5], H[10, 6], H[10, 7], H[10, 8], H[10, 9], H[10, 10], H[10, 11]], [H[11, 3], H[11, 4], H[11, 5], H[11, 6], H[11, 7], H[11, 8], H[11, 9], H[11, 10], H[11, 11]]])
    return project_pd(eH)


@ti.func
def point_triangle_ccd_broadphase(p0, t0, t1, t2, dHat):
    min_t = ti.min(ti.min(t0, t1), t2)
    max_t = ti.max(ti.max(t0, t1), t2)
    return (p0 < max_t + dHat).all() and (min_t - dHat < p0).all()


@ti.func
def moving_point_triangle_ccd_broadphase(p0, t0, t1, t2, dp0, dt0, dt1, dt2, dHat):
    max_p = ti.max(p0, p0 + dp0)
    min_p = ti.min(p0, p0 + dp0)
    max_t = ti.max(ti.max(t0, t0 + dt0), ti.max(ti.max(t1, t1 + dt1), ti.max(t2, t2 + dt2)))
    min_t = ti.min(ti.min(t0, t0 + dt0), ti.min(ti.min(t1, t1 + dt1), ti.min(t2, t2 + dt2)))
    return (min_p < max_t + dHat).all() and (min_t - dHat < max_p).all()


@ti.func
def edge_edge_ccd_broadphase(a0, a1, b0, b1, dHat):
    max_a = ti.max(a0, a1)
    min_a = ti.min(a0, a1)
    max_b = ti.max(b0, b1)
    min_b = ti.min(b0, b1)
    return (min_a < max_b + dHat).all() and (min_b - dHat < max_a).all()


@ti.func
def moving_edge_edge_ccd_broadphase(a0, a1, b0, b1, da0, da1, db0, db1, dHat):
    max_a = ti.max(ti.max(a0, a0 + da0), ti.max(a1, a1 + da1))
    min_a = ti.min(ti.min(a0, a0 + da0), ti.min(a1, a1 + da1))
    max_b = ti.max(ti.max(b0, b0 + db0), ti.max(b1, b1 + db1))
    min_b = ti.min(ti.min(b0, b0 + db0), ti.min(b1, b1 + db1))
    return (min_a < max_b + dHat).all() and (min_b - dHat < max_a).all()


@ti.func
def point_edge_ccd_broadphase(x0, x1, x2, dHat):
    min_e = ti.min(x1, x2)
    max_e = ti.max(x1, x2)
    return (x0 < max_e + dHat).all() and (min_e - dHat < x0).all()


@ti.func
def moving_point_edge_ccd_broadphase(x0, x1, x2, d0, d1, d2, dHat):
    min_p = ti.min(x0, x0 + d0)
    max_p = ti.max(x0, x0 + d0)
    min_e = ti.min(ti.min(x1, x2), ti.min(x1 + d1, x2 + d2))
    max_e = ti.max(ti.max(x1, x2), ti.max(x1 + d1, x2 + d2))
    return (min_p < max_e + dHat).all() and (min_e - dHat < max_p).all()


@ti.func
def check_overlap(x0, x1, x2, d0, d1, d2, root):
    p0 = x0 + d0 * root
    e0 = x1 + d1 * root
    e1 = x2 + d2 * root
    e = e1 - e0
    ratio = e.dot(p0 - e0) / e.norm_sqr()
    return 0 <= ratio and ratio <= 1


@ti.func
def moving_point_edge_ccd(x0, x1, x2, d0, d1, d2, eta):
    toc = 1.0
    a = d0[0] * (d2[1] - d1[1]) + d0[1] * (d1[0] - d2[0]) + d2[0] * d1[1] - d2[1] * d1[0]
    b = x0[0] * (d2[1] - d1[1]) + d0[0] * (x2[1] - x1[1]) + d0[1] * (x1[0] - x2[0]) + x0[1] * (d1[0] - d2[0]) + d1[1] * x2[0] + d2[0] * x1[1] - d1[0] * x2[1] - d2[1] * x1[0]
    c = x0[0] * (x2[1] - x1[1]) + x0[1] * (x1[0] - x2[0]) + x2[0] * x1[1] - x2[1] * x1[0]
    if a == 0 and b == 0 and c == 0:
        if (x0 - x1).dot(d0 - d1) < 0:
            root = ti.sqrt((x0 - x1).norm_sqr() / (d0 - d1).norm_sqr())
            if root > 0 and root <= 1:
                toc = ti.min(toc, root * (1 - eta))
        if (x0 - x2).dot(d0 - d2) < 0:
            root = ti.sqrt((x0 - x2).norm_sqr() / (d0 - d2).norm_sqr())
            if root > 0 and root <= 1:
                toc = ti.min(toc, root * (1 - eta))
    else:
        if a == 0:
            if b != 0:
                root = -c / b
                if root > 0 and root <= 1:
                    if check_overlap(x0, x1, x2, d0, d1, d2, root):
                        toc = ti.min(toc, root * (1 - eta))
        else:
            delta = b * b - 4 * a * c
            if delta == 0:
                root = -b / (2 * a)
                if root > 0 and root <= 1:
                    if check_overlap(x0, x1, x2, d0, d1, d2, root):
                        toc = ti.min(toc, root * (1 - eta))
            elif delta > 0:
                if b > 0:
                    root = (-b - ti.sqrt(delta)) / (2 * a)
                    if root > 0 and root <= 1:
                        if check_overlap(x0, x1, x2, d0, d1, d2, root):
                            toc = ti.min(toc, root * (1 - eta))
                    root = 2 * c / (-b - ti.sqrt(delta))
                    if root > 0 and root <= 1:
                        if check_overlap(x0, x1, x2, d0, d1, d2, root):
                            toc = ti.min(toc, root * (1 - eta))
                else:
                    root = 2 * c / (-b + ti.sqrt(delta))
                    if root > 0 and root <= 1:
                        if check_overlap(x0, x1, x2, d0, d1, d2, root):
                            toc = ti.min(toc, root * (1 - eta))
                    root = (-b + ti.sqrt(delta)) / (2 * a)
                    if root > 0 and root <= 1:
                        if check_overlap(x0, x1, x2, d0, d1, d2, root):
                            toc = ti.min(toc, root * (1 - eta))
    return toc
