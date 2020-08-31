import taichi as ti
from distance import *


@ti.func
def ipc_overlap(p0, e0, e1):
    e = e1 - e0
    ratio = e.dot(p0 - e0) / e.norm_sqr()
    E = False
    if ratio < 0:
        if point_point_energy(p0, e0) < 1e-8:
            E = True
    elif ratio > 1:
        if point_point_energy(p0, e1) < 1e-8:
            E = True
    else:
        if point_edge_energy(p0, e0, e1) < 1e-8:
            E = True
    if E:
        print("ERIUEIUIDUFIUFDIFU")
    return E


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
