import taichi as ti
import numpy as np
from diff_test import *

@ti.func
def simplex_volume(v0, v1, v2):
    return (v1 - v0).cross(v2 - v0).norm() * 0.5

@ti.func
def area_weighted_normal(xp, xe0, xe1):
    e = xe1 - xe0
    mHat = xp - (xe0 + (xp - xe0).dot(e) / e.norm_sqr() * e)
    mHat *= (0.5 * e.norm() / mHat.norm())
    return mHat 

@ti.func
def simplex_volume_gradient(v0, v1, v2): 
    grad = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dv_dv0 = area_weighted_normal(v0, v1, v2)
    dv_dv1 = area_weighted_normal(v1, v2, v0)
    dv_dv2 = area_weighted_normal(v2, v0, v1)
    for d in ti.static(range(3)):
        grad[0 * 3 + d] = dv_dv0[d]
        grad[1 * 3 + d] = dv_dv1[d]
        grad[2 * 3 + d] = dv_dv2[d]
    return grad

@ti.kernel
def simplex_volume_numpy(v: ti.ext_arr()) -> ti.float64:
    v0_ti = ti.Vector([v[0], v[1], v[2]]) 
    v1_ti = ti.Vector([v[3], v[4], v[5]])
    v2_ti = ti.Vector([v[6], v[7], v[8]])
    return simplex_volume(v0_ti, v1_ti, v2_ti)

@ti.kernel
def simplex_volume_gradient_numpy(v: ti.ext_arr(), arr: ti.ext_arr()):
    v0_ti = ti.Vector([v[0], v[1], v[2]])
    v1_ti = ti.Vector([v[3], v[4], v[5]])
    v2_ti = ti.Vector([v[6], v[7], v[8]])
    grad = simplex_volume_gradient(v0_ti, v1_ti, v2_ti)
    for i in ti.static(range(9)):
        arr[i] = grad[i]

if __name__ == '__main__':
    v = np.random.random((9,))
    check_gradient(v, simplex_volume_numpy, simplex_volume_gradient_numpy)
    check_gradient(v, simplex_volume_numpy, simplex_volume_gradient_numpy)
    check_gradient(v, simplex_volume_numpy, simplex_volume_gradient_numpy)
    check_gradient(v, simplex_volume_numpy, simplex_volume_gradient_numpy)
    check_gradient(v, simplex_volume_numpy, simplex_volume_gradient_numpy)