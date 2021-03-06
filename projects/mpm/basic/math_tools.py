import taichi as ti
# from common.math.external_func import *


@ti.func
def extract_vec(v, idx: ti.template()):
    vec = ti.Matrix.zero(ti.get_runtime().default_fp, len(idx))
    for i, j in ti.static(enumerate(idx)):
        vec[i] = v[j]
    return vec


@ti.func
def extract_mat(m, idx: ti.template()):
    mat = ti.Matrix.zero(ti.get_runtime().default_fp, len(idx), len(idx))
    for i, j in ti.static(enumerate(idx)):
        for p, q in ti.static(enumerate(idx)):
            mat[i, p] = m[j, q]
    return mat


@ti.func
def fill_vec(v, idx: ti.template(), n: ti.template()):
    vec = ti.Matrix.zero(ti.get_runtime().default_fp, n)
    for i, j in ti.static(enumerate(idx)):
        vec[j] = v[i]
    return vec


@ti.func
def fill_mat(m, idx: ti.template(), n: ti.template()):
    mat = ti.Matrix.zero(ti.get_runtime().default_fp, n, n)
    for i, j in ti.static(enumerate(idx)):
        for p, q in ti.static(enumerate(idx)):
            mat[j, q] = m[i, p]
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


@ti.func
def cofactor(F):
    if ti.static(F.n == 2):
        return ti.Matrix([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]])
    else:
        return ti.Matrix([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]],
                          [F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1]],
                          [F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])

@ti.func
def svd(F):
    return ti.svd(F)

# @ti.func
# def svd(F):
#     if ti.static(F.n == 2):
#         F00, F01, F10, F11 = F[0, 0], F[0, 1], F[1, 0], F[1, 1]
#         U00, U01, U10, U11 = 0.0, 0.0, 0.0, 0.0
#         s00, s01, s10, s11 = 0.0, 0.0, 0.0, 0.0
#         V00, V01, V10, V11 = 0.0, 0.0, 0.0, 0.0
#         ti.external_func_call(func=so.svd_2,
#                               args=(F00, F01, F10, F11),
#                               outputs=(U00, U01, U10, U11, s00, s01, s10, s11, V00, V01, V10, V11))
#         return ti.Matrix([[U00, U01], [U10, U11]]), ti.Matrix([[s00, s01], [s10, s11]]), ti.Matrix([[V00, V01], [V10, V11]])
#     if ti.static(F.n == 3):
#         F00, F01, F02, F10, F11, F12, F20, F21, F22 = F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2]
#         U00, U01, U02, U10, U11, U12, U20, U21, U22 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#         s00, s01, s02, s10, s11, s12, s20, s21, s22 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#         V00, V01, V02, V10, V11, V12, V20, V21, V22 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#         ti.external_func_call(func=so.svd_3,
#                               args=(F00, F01, F02, F10, F11, F12, F20, F21, F22),
#                               outputs=(U00, U01, U02, U10, U11, U12, U20, U21, U22, s00, s01, s02, s10, s11, s12, s20, s21, s22, V00, V01, V02, V10, V11, V12, V20, V21, V22))
#         return ti.Matrix([[U00, U01, U02], [U10, U11, U12], [U20, U21, U22]]), ti.Matrix([[s00, s01, s02], [s10, s11, s12], [s20, s21, s22]]), ti.Matrix([[V00, V01, V02], [V10, V11, V12], [V20, V21, V22]])



@ti.func
def project_pd(F):
    if ti.static(F.n == 2):
        return make_pd(F)
    if ti.static(F.n == 3):
        U, Sig, V = svd(F)
        for i in range(3):
            if Sig[i,i] < 0:
                Sig[i,i] = 0.0
        return U @ Sig @ V.transpose()
#     if ti.static(F.n == 3):
#         in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8 = F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2]
#         out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#         ti.external_func_call(func=so.project_pd_3, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8))
#         return ti.Matrix([[out_0, out_1, out_2], [out_3, out_4, out_5], [out_6, out_7, out_8]])
#     if ti.static(F.n == 4):
#         in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[3, 0], F[3, 1], F[3, 2], F[3, 3]
#         out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#         ti.external_func_call(func=so.project_pd_4, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15))
#         return ti.Matrix([[out_0, out_1, out_2, out_3], [out_4, out_5, out_6, out_7], [out_8, out_9, out_10, out_11], [out_12, out_13, out_14, out_15]])
#     if ti.static(F.n == 6):
#         in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5]
#         out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#         ti.external_func_call(func=so.project_pd_6, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35))
#         return ti.Matrix([[out_0, out_1, out_2, out_3, out_4, out_5], [out_6, out_7, out_8, out_9, out_10, out_11], [out_12, out_13, out_14, out_15, out_16, out_17], [out_18, out_19, out_20, out_21, out_22, out_23], [out_24, out_25, out_26, out_27, out_28, out_29], [out_30, out_31, out_32, out_33, out_34, out_35]])
#     if ti.static(F.n == 9):
#         in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[0, 6], F[0, 7], F[0, 8], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[1, 6], F[1, 7], F[1, 8], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[2, 6], F[2, 7], F[2, 8], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[3, 6], F[3, 7], F[3, 8], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[4, 6], F[4, 7], F[4, 8], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5], F[5, 6], F[5, 7], F[5, 8], F[6, 0], F[6, 1], F[6, 2], F[6, 3], F[6, 4], F[6, 5], F[6, 6], F[6, 7], F[6, 8], F[7, 0], F[7, 1], F[7, 2], F[7, 3], F[7, 4], F[7, 5], F[7, 6], F[7, 7], F[7, 8], F[8, 0], F[8, 1], F[8, 2], F[8, 3], F[8, 4], F[8, 5], F[8, 6], F[8, 7], F[8, 8]
#         out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35, out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71, out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#         ti.external_func_call(func=so.project_pd_9, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35, out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71, out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80))
#         return ti.Matrix([[out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8], [out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17], [out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26], [out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35], [out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44], [out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53], [out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62], [out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71], [out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80]])
#     if ti.static(F.n == 12):
#         in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80, in_81, in_82, in_83, in_84, in_85, in_86, in_87, in_88, in_89, in_90, in_91, in_92, in_93, in_94, in_95, in_96, in_97, in_98, in_99, in_100, in_101, in_102, in_103, in_104, in_105, in_106, in_107, in_108, in_109, in_110, in_111, in_112, in_113, in_114, in_115, in_116, in_117, in_118, in_119, in_120, in_121, in_122, in_123, in_124, in_125, in_126, in_127, in_128, in_129, in_130, in_131, in_132, in_133, in_134, in_135, in_136, in_137, in_138, in_139, in_140, in_141, in_142, in_143 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[0, 6], F[0, 7], F[0, 8], F[0, 9], F[0, 10], F[0, 11], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[1, 6], F[1, 7], F[1, 8], F[1, 9], F[1, 10], F[1, 11], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[2, 6], F[2, 7], F[2, 8], F[2, 9], F[2, 10], F[2, 11], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[3, 6], F[3, 7], F[3, 8], F[3, 9], F[3, 10], F[3, 11], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[4, 6], F[4, 7], F[4, 8], F[4, 9], F[4, 10], F[4, 11], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5], F[5, 6], F[5, 7], F[5, 8], F[5, 9], F[5, 10], F[5, 11], F[6, 0], F[6, 1], F[6, 2], F[6, 3], F[6, 4], F[6, 5], F[6, 6], F[6, 7], F[6, 8], F[6, 9], F[6, 10], F[6, 11], F[7, 0], F[7, 1], F[7, 2], F[7, 3], F[7, 4], F[7, 5], F[7, 6], F[7, 7], F[7, 8], F[7, 9], F[7, 10], F[7, 11], F[8, 0], F[8, 1], F[8, 2], F[8, 3], F[8, 4], F[8, 5], F[8, 6], F[8, 7], F[8, 8], F[8, 9], F[8, 10], F[8, 11], F[9, 0], F[9, 1], F[9, 2], F[9, 3], F[9, 4], F[9, 5], F[9, 6], F[9, 7], F[9, 8], F[9, 9], F[9, 10], F[9, 11], F[10, 0], F[10, 1], F[10, 2], F[10, 3], F[10, 4], F[10, 5], F[10, 6], F[10, 7], F[10, 8], F[10, 9], F[10, 10], F[10, 11], F[11, 0], F[11, 1], F[11, 2], F[11, 3], F[11, 4], F[11, 5], F[11, 6], F[11, 7], F[11, 8], F[11, 9], F[11, 10], F[11, 11]
#         out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35, out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71, out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80, out_81, out_82, out_83, out_84, out_85, out_86, out_87, out_88, out_89, out_90, out_91, out_92, out_93, out_94, out_95, out_96, out_97, out_98, out_99, out_100, out_101, out_102, out_103, out_104, out_105, out_106, out_107, out_108, out_109, out_110, out_111, out_112, out_113, out_114, out_115, out_116, out_117, out_118, out_119, out_120, out_121, out_122, out_123, out_124, out_125, out_126, out_127, out_128, out_129, out_130, out_131, out_132, out_133, out_134, out_135, out_136, out_137, out_138, out_139, out_140, out_141, out_142, out_143 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#         ti.external_func_call(func=so.project_pd_12, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80, in_81, in_82, in_83, in_84, in_85, in_86, in_87, in_88, in_89, in_90, in_91, in_92, in_93, in_94, in_95, in_96, in_97, in_98, in_99, in_100, in_101, in_102, in_103, in_104, in_105, in_106, in_107, in_108, in_109, in_110, in_111, in_112, in_113, in_114, in_115, in_116, in_117, in_118, in_119, in_120, in_121, in_122, in_123, in_124, in_125, in_126, in_127, in_128, in_129, in_130, in_131, in_132, in_133, in_134, in_135, in_136, in_137, in_138, in_139, in_140, in_141, in_142, in_143), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35, out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71, out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80, out_81, out_82, out_83, out_84, out_85, out_86, out_87, out_88, out_89, out_90, out_91, out_92, out_93, out_94, out_95, out_96, out_97, out_98, out_99, out_100, out_101, out_102, out_103, out_104, out_105, out_106, out_107, out_108, out_109, out_110, out_111, out_112, out_113, out_114, out_115, out_116, out_117, out_118, out_119, out_120, out_121, out_122, out_123, out_124, out_125, out_126, out_127, out_128, out_129, out_130, out_131, out_132, out_133, out_134, out_135, out_136, out_137, out_138, out_139, out_140, out_141, out_142, out_143))
#         return ti.Matrix([[out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11], [out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23], [out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35], [out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47], [out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59], [out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71], [out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80, out_81, out_82, out_83], [out_84, out_85, out_86, out_87, out_88, out_89, out_90, out_91, out_92, out_93, out_94, out_95], [out_96, out_97, out_98, out_99, out_100, out_101, out_102, out_103, out_104, out_105, out_106, out_107], [out_108, out_109, out_110, out_111, out_112, out_113, out_114, out_115, out_116, out_117, out_118, out_119], [out_120, out_121, out_122, out_123, out_124, out_125, out_126, out_127, out_128, out_129, out_130, out_131], [out_132, out_133, out_134, out_135, out_136, out_137, out_138, out_139, out_140, out_141, out_142, out_143]])


# @ti.func
# def solve(F, rhs):
#     if ti.static(F.n == 2):
#         in_0, in_1, in_2, in_3, in_4, in_5 = F[0, 0], F[0, 1], F[1, 0], F[1, 1], rhs[0], rhs[1]
#         out_0, out_1 = 0.0, 0.0
#         ti.external_func_call(func=so.solve_2, args=(in_0, in_1, in_2, in_3, in_4, in_5), outputs=(out_0, out_1))
#         return ti.Vector([out_0, out_1])
#     if ti.static(F.n == 3):
#         in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11 = F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2], rhs[0], rhs[1], rhs[2]
#         out_0, out_1, out_2 = 0.0, 0.0, 0.0
#         ti.external_func_call(func=so.solve_3, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11), outputs=(out_0, out_1, out_2))
#         return ti.Vector([out_0, out_1, out_2])
#     if ti.static(F.n == 4):
#         in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[3, 0], F[3, 1], F[3, 2], F[3, 3], rhs[0], rhs[1], rhs[2], rhs[3]
#         out_0, out_1, out_2, out_3 = 0.0, 0.0, 0.0, 0.0
#         ti.external_func_call(func=so.solve_4, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19), outputs=(out_0, out_1, out_2, out_3))
#         return ti.Vector([out_0, out_1, out_2, out_3])
#     if ti.static(F.n == 6):
#         in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5], rhs[0], rhs[1], rhs[2], rhs[3], rhs[4], rhs[5]
#         out_0, out_1, out_2, out_3, out_4, out_5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#         ti.external_func_call(func=so.solve_6, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41), outputs=(out_0, out_1, out_2, out_3, out_4, out_5))
#         return ti.Vector([out_0, out_1, out_2, out_3, out_4, out_5])
#     if ti.static(F.n == 9):
#         in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80, in_81, in_82, in_83, in_84, in_85, in_86, in_87, in_88, in_89 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[0, 6], F[0, 7], F[0, 8], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[1, 6], F[1, 7], F[1, 8], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[2, 6], F[2, 7], F[2, 8], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[3, 6], F[3, 7], F[3, 8], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[4, 6], F[4, 7], F[4, 8], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5], F[5, 6], F[5, 7], F[5, 8], F[6, 0], F[6, 1], F[6, 2], F[6, 3], F[6, 4], F[6, 5], F[6, 6], F[6, 7], F[6, 8], F[7, 0], F[7, 1], F[7, 2], F[7, 3], F[7, 4], F[7, 5], F[7, 6], F[7, 7], F[7, 8], F[8, 0], F[8, 1], F[8, 2], F[8, 3], F[8, 4], F[8, 5], F[8, 6], F[8, 7], F[8, 8], rhs[0], rhs[1], rhs[2], rhs[3], rhs[4], rhs[5], rhs[6], rhs[7], rhs[8]
#         out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#         ti.external_func_call(func=so.solve_9, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80, in_81, in_82, in_83, in_84, in_85, in_86, in_87, in_88, in_89), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8))
#         return ti.Vector([out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8])


# @ti.func
# def get_smallest_positive_real_cubic_root(x1, x2, x3, x4, p1, p2, p3, p4, eta):
#     v01 = x2 - x1
#     v02 = x3 - x1
#     v03 = x4 - x1
#     p01 = p2 - p1
#     p02 = p3 - p1
#     p03 = p4 - p1
#     p01xp02 = p01.cross(p02)
#     v01xp02pp01xv02 = v01.cross(p02) + p01.cross(v02)
#     v01xv02 = v01.cross(v02)
#     a = p03.dot(p01xp02)
#     b = v03.dot(p01xp02) + p03.dot(v01xp02pp01xv02)
#     c = p03.dot(v01xv02) + v03.dot(v01xp02pp01xv02)
#     d = v03.dot(v01xv02)
#     ret = 0.0
#     tol = 1e-8
#     ti.external_func_call(func=so.get_smallest_positive_real_cubic_root,
#                           args=(a, b, c, d, tol),
#                           outputs=(ret,))
#     ret = 1.0 if ret < 0.0 else ret * (1.0 - eta)
#     return ret
