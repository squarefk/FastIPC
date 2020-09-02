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
def project_pd_3(F):
    in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8 = F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2]
    out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ti.external_func_call(func=so.project_pd_3, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8))
    return ti.Matrix([[out_0, out_1, out_2], [out_3, out_4, out_5], [out_6, out_7, out_8]])
@ti.func
def project_pd_6(F):
    in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5]
    out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ti.external_func_call(func=so.project_pd_6, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35))
    return ti.Matrix([[out_0, out_1, out_2, out_3, out_4, out_5], [out_6, out_7, out_8, out_9, out_10, out_11], [out_12, out_13, out_14, out_15, out_16, out_17], [out_18, out_19, out_20, out_21, out_22, out_23], [out_24, out_25, out_26, out_27, out_28, out_29], [out_30, out_31, out_32, out_33, out_34, out_35]])
@ti.func
def project_pd_9(F):
    in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[0, 6], F[0, 7], F[0, 8], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[1, 6], F[1, 7], F[1, 8], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[2, 6], F[2, 7], F[2, 8], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[3, 6], F[3, 7], F[3, 8], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[4, 6], F[4, 7], F[4, 8], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5], F[5, 6], F[5, 7], F[5, 8], F[6, 0], F[6, 1], F[6, 2], F[6, 3], F[6, 4], F[6, 5], F[6, 6], F[6, 7], F[6, 8], F[7, 0], F[7, 1], F[7, 2], F[7, 3], F[7, 4], F[7, 5], F[7, 6], F[7, 7], F[7, 8], F[8, 0], F[8, 1], F[8, 2], F[8, 3], F[8, 4], F[8, 5], F[8, 6], F[8, 7], F[8, 8]
    out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35, out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71, out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ti.external_func_call(func=so.project_pd_9, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35, out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71, out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80))
    return ti.Matrix([[out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8], [out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17], [out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26], [out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35], [out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44], [out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53], [out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62], [out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71], [out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80]])
@ti.func
def inverse_6(F):
    in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5]
    out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ti.external_func_call(func=so.inverse_6, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35))
    return ti.Matrix([[out_0, out_1, out_2, out_3, out_4, out_5], [out_6, out_7, out_8, out_9, out_10, out_11], [out_12, out_13, out_14, out_15, out_16, out_17], [out_18, out_19, out_20, out_21, out_22, out_23], [out_24, out_25, out_26, out_27, out_28, out_29], [out_30, out_31, out_32, out_33, out_34, out_35]])
@ti.func
def inverse_9(F):
    in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[0, 6], F[0, 7], F[0, 8], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[1, 6], F[1, 7], F[1, 8], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[2, 6], F[2, 7], F[2, 8], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[3, 6], F[3, 7], F[3, 8], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[4, 6], F[4, 7], F[4, 8], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5], F[5, 6], F[5, 7], F[5, 8], F[6, 0], F[6, 1], F[6, 2], F[6, 3], F[6, 4], F[6, 5], F[6, 6], F[6, 7], F[6, 8], F[7, 0], F[7, 1], F[7, 2], F[7, 3], F[7, 4], F[7, 5], F[7, 6], F[7, 7], F[7, 8], F[8, 0], F[8, 1], F[8, 2], F[8, 3], F[8, 4], F[8, 5], F[8, 6], F[8, 7], F[8, 8]
    out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35, out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71, out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ti.external_func_call(func=so.inverse_9, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35, out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71, out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80))
    return ti.Matrix([[out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8], [out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17], [out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26], [out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35], [out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44], [out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53], [out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62], [out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71], [out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80]])



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