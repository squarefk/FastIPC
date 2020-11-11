import taichi as ti
from common.math.distance import *
from common.math.math_tools import *
from common.math.external_func import *


###########################################
# PP_{energy, gradient, hessian}
# PE_{energy, gradient, hessian}
# PT_{energy, gradient, hessian}
# EE_{energy, gradient, hessian}
# EEM_{energy, gradient, hessian}
# PPM_{energy, gradient, hessian}
# PEM_{energy, gradient, hessian}
# point_triangle_ccd_broadphase
# moving_point_triangle_ccd_broadphase
# edge_edge_ccd_broadphase
# moving_edge_edge_ccd_broadphase
# point_edge_ccd_broadphase
# moving_point_edge_ccd_broadphase
# point_triangle_ccd
# edge_edge_ccd
# point_edge_ccd
# point_inside_triangle
# segment_intersect_triangle
###########################################

COMMON_MATH_IPC_DISTANCE_THRESHOLD = 1e-16

@ti.func
def PP_energy(p0, p1, dHat2, kappa):
    if ti.static(p0.n == 2):
        dist2 = PP_2D_E(p0, p1)
        if dist2 < COMMON_MATH_IPC_DISTANCE_THRESHOLD:
            print('ERROR PP', dist2 * 1e16, 'e-16')
        return barrier_E(dist2, dHat2, kappa)
    else:
        dist2 = PP_3D_E(p0, p1)
        if dist2 < COMMON_MATH_IPC_DISTANCE_THRESHOLD:
            print('ERROR PP', dist2 * 1e16, 'e-16')
        return barrier_E(dist2, dHat2, kappa)
@ti.func
def PP_gradient(p0, p1, dHat2, kappa):
    if ti.static(p0.n == 2):
        dist2 = PP_2D_E(p0, p1)
        dist2g = PP_2D_g(p0, p1)
        bg = barrier_g(dist2, dHat2, kappa)
        return bg * dist2g
    else:
        dist2 = PP_3D_E(p0, p1)
        dist2g = PP_3D_g(p0, p1)
        bg = barrier_g(dist2, dHat2, kappa)
        return bg * dist2g
@ti.func
def PP_hessian(p0, p1, dHat2, kappa):
    if ti.static(p0.n == 2):
        dist2 = PP_2D_E(p0, p1)
        dist2g = PP_2D_g(p0, p1)
        bg = barrier_g(dist2, dHat2, kappa)
        return barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_2D_H(p0, p1)
    else:
        dist2 = PP_3D_E(p0, p1)
        dist2g = PP_3D_g(p0, p1)
        bg = barrier_g(dist2, dHat2, kappa)
        return barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_3D_H(p0, p1)
@ti.func
def PP_g_and_H(p0, p1, dHat2, kappa):
    if ti.static(p0.n == 2):
        dist2 = PP_2D_E(p0, p1)
        dist2g = PP_2D_g(p0, p1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_2D_H(p0, p1)
        return g, project_pd(H)
    else:
        dist2 = PP_3D_E(p0, p1)
        dist2g = PP_3D_g(p0, p1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_3D_H(p0, p1)
        return g, project_pd(H)


@ti.func
def PE_energy(p, e0, e1, dHat2, kappa):
    if ti.static(p.n == 2):
        dist2 = PE_2D_E(p, e0, e1)
        if dist2 < COMMON_MATH_IPC_DISTANCE_THRESHOLD:
            print('ERROR PE', dist2 * 1e16, 'e-16')
        return barrier_E(dist2, dHat2, kappa)
    else:
        dist2 = PE_3D_E(p, e0, e1)
        if dist2 < COMMON_MATH_IPC_DISTANCE_THRESHOLD:
            print('ERROR PE', dist2 * 1e16, 'e-16')
        return barrier_E(dist2, dHat2, kappa)
@ti.func
def PE_gradient(p, e0, e1, dHat2, kappa):
    if ti.static(p.n == 2):
        dist2 = PE_2D_E(p, e0, e1)
        dist2g = PE_2D_g(p, e0, e1)
        bg = barrier_g(dist2, dHat2, kappa)
        return bg * dist2g
    else:
        dist2 = PE_3D_E(p, e0, e1)
        dist2g = PE_3D_g(p, e0, e1)
        bg = barrier_g(dist2, dHat2, kappa)
        return bg * dist2g
@ti.func
def PE_hessian(p, e0, e1, dHat2, kappa):
    if ti.static(p.n == 2):
        dist2 = PE_2D_E(p, e0, e1)
        dist2g = PE_2D_g(p, e0, e1)
        bg = barrier_g(dist2, dHat2, kappa)
        return barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_2D_H(p, e0, e1)
    else:
        dist2 = PE_3D_E(p, e0, e1)
        dist2g = PE_3D_g(p, e0, e1)
        bg = barrier_g(dist2, dHat2, kappa)
        return barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_3D_H(p, e0, e1)
@ti.func
def PE_g_and_H(p, e0, e1, dHat2, kappa):
    if ti.static(p.n == 2):
        dist2 = PE_2D_E(p, e0, e1)
        dist2g = PE_2D_g(p, e0, e1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_2D_H(p, e0, e1)
        return g, project_pd(H)
    else:
        dist2 = PE_3D_E(p, e0, e1)
        dist2g = PE_3D_g(p, e0, e1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_3D_H(p, e0, e1)
        return g, project_pd(H)


@ti.func
def PT_energy(p, t0, t1, t2, dHat2, kappa):
    dist2 = PT_3D_E(p, t0, t1, t2)
    if dist2 < COMMON_MATH_IPC_DISTANCE_THRESHOLD:
        print('ERROR PT', dist2 * 1e16, 'e-16')
    return barrier_E(dist2, dHat2, kappa)
@ti.func
def PT_gradient(p, t0, t1, t2, dHat2, kappa):
    dist2 = PT_3D_E(p, t0, t1, t2)
    dist2g = PT_3D_g(p, t0, t1, t2)
    bg = barrier_g(dist2, dHat2, kappa)
    return bg * dist2g
@ti.func
def PT_hessian(p, t0, t1, t2, dHat2, kappa):
    dist2 = PT_3D_E(p, t0, t1, t2)
    dist2g = PT_3D_g(p, t0, t1, t2)
    bg = barrier_g(dist2, dHat2, kappa)
    return barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PT_3D_H(p, t0, t1, t2)
@ti.func
def PT_g_and_H(p, t0, t1, t2, dHat2, kappa):
    dist2 = PT_3D_E(p, t0, t1, t2)
    dist2g = PT_3D_g(p, t0, t1, t2)
    bg = barrier_g(dist2, dHat2, kappa)
    g = bg * dist2g
    H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PT_3D_H(p, t0, t1, t2)
    return g, project_pd(H)


@ti.func
def EE_energy(a0, a1, b0, b1, dHat2, kappa):
    dist2 = EE_3D_E(a0, a1, b0, b1)
    if dist2 < COMMON_MATH_IPC_DISTANCE_THRESHOLD:
        print('ERROR EE', dist2 * 1e16, 'e-16')
    return barrier_E(dist2, dHat2, kappa)
@ti.func
def EE_gradient(a0, a1, b0, b1, dHat2, kappa):
    dist2 = EE_3D_E(a0, a1, b0, b1)
    dist2g = EE_3D_g(a0, a1, b0, b1)
    bg = barrier_g(dist2, dHat2, kappa)
    return bg * dist2g
@ti.func
def EE_hessian(a0, a1, b0, b1, dHat2, kappa):
    dist2 = EE_3D_E(a0, a1, b0, b1)
    dist2g = EE_3D_g(a0, a1, b0, b1)
    bg = barrier_g(dist2, dHat2, kappa)
    return barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * EE_3D_H(a0, a1, b0, b1)
@ti.func
def EE_g_and_H(a0, a1, b0, b1, dHat2, kappa):
    dist2 = EE_3D_E(a0, a1, b0, b1)
    dist2g = EE_3D_g(a0, a1, b0, b1)
    bg = barrier_g(dist2, dHat2, kappa)
    g = bg * dist2g
    H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * EE_3D_H(a0, a1, b0, b1)
    return g, project_pd(H)


@ti.func
def EEM_energy(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = EE_3D_E(a0, a1, b0, b1)
    if dist2 < COMMON_MATH_IPC_DISTANCE_THRESHOLD:
        print('ERROR EEM', dist2 * 1e16, 'e-16')
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
    return lg * M + b * Mg
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
    return lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
@ti.func
def EEM_g_and_H(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = EE_3D_E(a0, a1, b0, b1)
    dist2g = EE_3D_g(a0, a1, b0, b1)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    lg = bg * dist2g
    lH = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * EE_3D_H(a0, a1, b0, b1)
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    g = lg * M + b * Mg
    H = lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
    return g, project_pd(H)


@ti.func
def PPM_energy(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PP_3D_E(a0, b0)
    if dist2 < COMMON_MATH_IPC_DISTANCE_THRESHOLD:
        print('ERROR PPM', dist2 * 1e16, 'e-16')
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
    return lg * M + b * Mg
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
    return lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
@ti.func
def PPM_g_and_H(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PP_3D_E(a0, b0)
    dist2g = PP_3D_g(a0, b0)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    lg = fill_vec(bg * dist2g, [0, 1, 2, 6, 7, 8])
    lH = fill_mat(barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_3D_H(a0, b0), [0, 1, 2, 6, 7, 8])
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    g = lg * M + b * Mg
    H = lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
    return g, project_pd(H)


@ti.func
def PEM_energy(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PE_3D_E(a0, b0, b1)
    if dist2 < COMMON_MATH_IPC_DISTANCE_THRESHOLD:
        print('ERROR PEM', dist2 * 1e16, 'e-16')
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
    return lg * M + b * Mg
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
    return lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
@ti.func
def PEM_g_and_H(a0, a1, b0, b1, _a0, _a1, _b0, _b1, dHat2, kappa):
    eps_x = M_threshold(_a0, _a1, _b0, _b1)
    dist2 = PE_3D_E(a0, b0, b1)
    dist2g = PE_3D_g(a0, b0, b1)
    b = barrier_E(dist2, dHat2, kappa)
    bg = barrier_g(dist2, dHat2, kappa)
    lg = fill_vec(bg * dist2g, [0, 1, 2, 6, 7, 8, 9, 10, 11])
    lH = fill_mat(barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_3D_H(a0, b0, b1), [0, 1, 2, 6, 7, 8, 9, 10, 11])
    M = M_E(a0, a1, b0, b1, eps_x)
    Mg = M_g(a0, a1, b0, b1, eps_x)
    g = lg * M + b * Mg
    H = lH * M + lg.outer_product(Mg) + Mg.outer_product(lg) + b * M_H(a0, a1, b0, b1, eps_x)
    return g, project_pd(H)


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


@ti.func
def check_overlap(x0, x1, x2, d0, d1, d2, root):
    p0 = x0 + d0 * root
    e0 = x1 + d1 * root
    e1 = x2 + d2 * root
    e = e1 - e0
    ratio = e.dot(p0 - e0) / e.norm_sqr()
    return 0 <= ratio and ratio <= 1
@ti.func
def point_edge_ccd_impl(x0, x1, x2, d0, d1, d2, eta):
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
@ti.func
def point_edge_ccd(x0, x1, x2, d0, d1, d2, eta):
    dist2_cur = PE_dist2(x0, x1, x2, PE_type(x0, x1, x2))
    maxDispMag2 = max(d0.norm_sqr(), d1.norm_sqr(), d2.norm_sqr())
    toc = 1.0
    if maxDispMag2 > 0:
        tocLowerBound = (1 - eta) * ti.sqrt(dist2_cur) / (2 * ti.sqrt(maxDispMag2))
        if tocLowerBound <= 1:
            toc = point_edge_ccd_impl(x0, x1, x2, d0, d1, d2, eta)
            toc = max(toc, tocLowerBound)
    return toc


# @ti.func
# def point_inside_triangle(P, A, B, C):
#     v0 = C - A
#     v1 = B - A
#     v2 = P - A
#     # Compute dot products
#     dot00 = v0.dot(v0)
#     dot01 = v0.dot(v1)
#     dot02 = v0.dot(v2)
#     dot11 = v1.dot(v1)
#     dot12 = v1.dot(v2)
#     # Compute barycentric coordinates
#     invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
#     u = (dot11 * dot02 - dot01 * dot12) * invDenom
#     v = (dot00 * dot12 - dot01 * dot02) * invDenom
#     # Check if point is in triangle
#     return u >= 0 and v >= 0 and u + v < 1
@ti.func
def point_inside_triangle(P, A, B, C):
    d1 = (P - B).cross(A - B)
    d2 = (P - C).cross(B - C)
    d3 = (P - A).cross(C - A)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


@ti.func
def segment_intersect_triangle(P, Q, A, B, C):
    RLen = (Q - P).norm()
    RDir = (Q - P) / RLen
    ROrigin = P
    E1 = B - A
    E2 = C - A
    N = E1.cross(E2)
    det = -RDir.dot(N)
    invdet = 1.0 / det
    AO  = ROrigin - A
    DAO = AO.cross(RDir)
    u = E2.dot(DAO) * invdet
    v = -E1.dot(DAO) * invdet
    t = AO.dot(N) * invdet
    return det >= 1e-12 and t >= 0.0 and u >= 0.0 and v >= 0.0 and (u+v) <= 1.0 and t <= RLen


@ti.func
def line_intersection_test(p0, p1, p2, p3):
    s1_x = float(p1[0] - p0[0])
    s1_y = float(p1[1] - p0[1])
    s2_x = float(p3[0] - p2[0])
    s2_y = float(p3[1] - p2[1])
    s = float(-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / (-s2_x * s1_y + s1_x * s2_y)
    t = float( s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / (-s2_x * s1_y + s1_x * s2_y)
    return 0 <= s <= 1 and 0 <= t <= 1


@ti.func
def segment_intersect_triangle_2D(e0, e1, t0, t1, t2):
    return line_intersection_test(e0, e1, t0, t1) or line_intersection_test(e0, e1, t0, t2) or line_intersection_test(e0, e1, t1, t2)
