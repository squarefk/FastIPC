import taichi as ti
from distance import *

real = ti.f64
ti.init(arch=ti.cpu, default_fp=real)


@ti.kernel
def test_barrier():
    print("===================== test =====================")
    h = 1.0
    a = ti.random()
    step = 1.0
    for i in range(20):
        h /= 2
        aa = a + step * h
        err1 = (barrier_E(a, 1, 1) - barrier_E(aa, 1, 1)) / h + (barrier_g(a, 1, 1) + barrier_g(aa, 1, 1)) * step / 2
        err2 = (barrier_g(a, 1, 1) - barrier_g(aa, 1, 1)) / h + (barrier_H(a, 1, 1) + barrier_H(aa, 1, 1)) * step / 2
        print(h, err1, err2, ti.log(abs(err1)), ti.log(abs(err2)))


@ti.func
def rand_v(dim: ti.template()):
    v = ti.Matrix.zero(real, dim)
    for d in ti.static(range(dim)):
        v[d] = ti.random()
    return v

@ti.kernel
def test_PT_EE_PE_PP_M():
    print("===================== test =====================")
    h = 1.0
    a = rand_v(3)
    b = rand_v(3)
    c = rand_v(3)
    d = rand_v(3)
    step = rand_v(12)
    step /= step.norm()
    for i in range(20):
        h /= 2
        aa = a + ti.Vector([step[0], step[1], step[2]]) * h
        bb = b + ti.Vector([step[3], step[4], step[5]]) * h
        cc = c + ti.Vector([step[6], step[7], step[8]]) * h
        dd = d + ti.Vector([step[9], step[10], step[11]]) * h
        step12 = step
        step9 = ti.Vector([step[0], step[1], step[2], step[3], step[4], step[5], step[6], step[7], step[8]])
        step6 = ti.Vector([step[0], step[1], step[2], step[3], step[4], step[5]])
        # PT0, PT1, EE0, EE1, PE0, PE1, PP0, PP1 = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        # PT0 = (PT_3D_E(a, b, c, d) - PT_3D_E(aa, bb, cc, dd)) / h + (PT_3D_g(a, b, c, d) + PT_3D_g(aa, bb, cc, dd)).dot(step12) / 2
        # PT1 = ((PT_3D_g(a, b, c, d) - PT_3D_g(aa, bb, cc, dd)) / h + (PT_3D_H(a, b, c, d) + PT_3D_H(aa, bb, cc, dd)) @ step12 / 2).norm_sqr()
        # EE0 = (EE_3D_E(a, b, c, d) - EE_3D_E(aa, bb, cc, dd)) / h + (EE_3D_g(a, b, c, d) + EE_3D_g(aa, bb, cc, dd)).dot(step12) / 2
        # EE1 = ((EE_3D_g(a, b, c, d) - EE_3D_g(aa, bb, cc, dd)) / h + (EE_3D_H(a, b, c, d) + EE_3D_H(aa, bb, cc, dd)) @ step12 / 2).norm_sqr()
        # PE0 = (PE_3D_E(a, b, c) - PE_3D_E(aa, bb, cc)) / h + (PE_3D_g(a, b, c) + PE_3D_g(aa, bb, cc)).dot(step9) / 2
        # PE1 = ((PE_3D_g(a, b, c) - PE_3D_g(aa, bb, cc)) / h + (PE_3D_H(a, b, c) + PE_3D_H(aa, bb, cc)) @ step9 / 2).norm_sqr()
        # PP0 = (PP_3D_E(a, b) - PP_3D_E(aa, bb)) / h + (PP_3D_g(a, b) + PP_3D_g(aa, bb)).dot(step6) / 2
        # PP1 = ((PP_3D_g(a, b) - PP_3D_g(aa, bb)) / h + (PP_3D_H(a, b) + PP_3D_H(aa, bb)) @ step6 / 2).norm_sqr()
        # print(i, ti.log(abs(PT0)), ti.log(abs(PT1)), ti.log(abs(EE0)), ti.log(abs(EE1)), ti.log(abs(PE0)), ti.log(abs(PE1)), ti.log(abs(PP0)), ti.log(abs(PP1)))
        M0, M1 = 1.0, 1.0
        eps_x = 0.1
        M0 = (M_E(a, b, c, d, eps_x) - M_E(aa, bb, cc, dd, eps_x)) / h + (M_g(a, b, c, d, eps_x) + M_g(aa, bb, cc, dd, eps_x)).dot(step12) / 2
        M1 = ((M_g(a, b, c, d, eps_x) - M_g(aa, bb, cc, dd, eps_x)) / h + (M_H(a, b, c, d, eps_x) + M_H(aa, bb, cc, dd, eps_x)) @ step12 / 2).norm_sqr()
        print(i, ti.log(abs(M0)), ti.log(abs(M1)))


if __name__ == "__main__":
    test_barrier()
    test_PT_EE_PE_PP_M()