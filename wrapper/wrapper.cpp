#include "SINGULAR_VALUE_DECOMPOSITION.h"
#include "VECTOR.h"
#include "UTILS.h"
#include "EVCTCD/CTCD.h"
#include <Eigen/Eigen>
#include <complex>
#include <cstdlib>
using namespace std;

extern "C" {
    void svd(double F00, double F01, double F10, double F11,
             double* U00, double* U01, double* U10, double* U11,
             double* s00, double* s01, double* s10, double* s11,
             double* V00, double* V01, double* V10, double* V11)
    {
        JGSL::MATRIX<double, 2> F;
        F(0, 0) = F00; F(0, 1) = F01; F(1, 0) = F10; F(1, 1) = F11;
        JGSL::MATRIX<double, 2> U(1), V(1);
        JGSL::VECTOR<double, 2> sigma;
        JGSL::Singular_Value_Decomposition(F, U, sigma, V);
        U00[0] = U(0, 0),  U01[0] = U(0, 1), U10[0] = U(1, 0), U11[0] = U(1, 1);
        s00[0] = sigma(0), s01[0] = 0,      s10[0] = 0,       s11[0] = sigma(1);
        V00[0] = V(0, 0),  V01[0] = V(0, 1), V10[0] = V(1, 0), V11[0] = V(1, 1);
    }

    void project_pd_3(double in_0, double in_1, double in_2, double in_3, double in_4, double in_5, double in_6, double in_7, double in_8, double* out_0, double* out_1, double* out_2, double* out_3, double* out_4, double* out_5, double* out_6, double* out_7, double* out_8)
    {
        Eigen::Matrix<double, 3, 3> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(1, 0) = in_3; F(1, 1) = in_4; F(1, 2) = in_5; F(2, 0) = in_6; F(2, 1) = in_7; F(2, 2) = in_8;
        JGSL::makePD(F);
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(1, 0); out_4[0] = F(1, 1); out_5[0] = F(1, 2); out_6[0] = F(2, 0); out_7[0] = F(2, 1); out_8[0] = F(2, 2);
    }

    void project_pd_6(double in_0, double in_1, double in_2, double in_3, double in_4, double in_5, double in_6, double in_7, double in_8, double in_9, double in_10, double in_11, double in_12, double in_13, double in_14, double in_15, double in_16, double in_17, double in_18, double in_19, double in_20, double in_21, double in_22, double in_23, double in_24, double in_25, double in_26, double in_27, double in_28, double in_29, double in_30, double in_31, double in_32, double in_33, double in_34, double in_35, double* out_0, double* out_1, double* out_2, double* out_3, double* out_4, double* out_5, double* out_6, double* out_7, double* out_8, double* out_9, double* out_10, double* out_11, double* out_12, double* out_13, double* out_14, double* out_15, double* out_16, double* out_17, double* out_18, double* out_19, double* out_20, double* out_21, double* out_22, double* out_23, double* out_24, double* out_25, double* out_26, double* out_27, double* out_28, double* out_29, double* out_30, double* out_31, double* out_32, double* out_33, double* out_34, double* out_35)
    {
        Eigen::Matrix<double, 6, 6> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(0, 4) = in_4; F(0, 5) = in_5; F(1, 0) = in_6; F(1, 1) = in_7; F(1, 2) = in_8; F(1, 3) = in_9; F(1, 4) = in_10; F(1, 5) = in_11; F(2, 0) = in_12; F(2, 1) = in_13; F(2, 2) = in_14; F(2, 3) = in_15; F(2, 4) = in_16; F(2, 5) = in_17; F(3, 0) = in_18; F(3, 1) = in_19; F(3, 2) = in_20; F(3, 3) = in_21; F(3, 4) = in_22; F(3, 5) = in_23; F(4, 0) = in_24; F(4, 1) = in_25; F(4, 2) = in_26; F(4, 3) = in_27; F(4, 4) = in_28; F(4, 5) = in_29; F(5, 0) = in_30; F(5, 1) = in_31; F(5, 2) = in_32; F(5, 3) = in_33; F(5, 4) = in_34; F(5, 5) = in_35;
        JGSL::makePD(F);
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(0, 4); out_5[0] = F(0, 5); out_6[0] = F(1, 0); out_7[0] = F(1, 1); out_8[0] = F(1, 2); out_9[0] = F(1, 3); out_10[0] = F(1, 4); out_11[0] = F(1, 5); out_12[0] = F(2, 0); out_13[0] = F(2, 1); out_14[0] = F(2, 2); out_15[0] = F(2, 3); out_16[0] = F(2, 4); out_17[0] = F(2, 5); out_18[0] = F(3, 0); out_19[0] = F(3, 1); out_20[0] = F(3, 2); out_21[0] = F(3, 3); out_22[0] = F(3, 4); out_23[0] = F(3, 5); out_24[0] = F(4, 0); out_25[0] = F(4, 1); out_26[0] = F(4, 2); out_27[0] = F(4, 3); out_28[0] = F(4, 4); out_29[0] = F(4, 5); out_30[0] = F(5, 0); out_31[0] = F(5, 1); out_32[0] = F(5, 2); out_33[0] = F(5, 3); out_34[0] = F(5, 4); out_35[0] = F(5, 5);
    }

    void project_pd_9(double in_0, double in_1, double in_2, double in_3, double in_4, double in_5, double in_6, double in_7, double in_8, double in_9, double in_10, double in_11, double in_12, double in_13, double in_14, double in_15, double in_16, double in_17, double in_18, double in_19, double in_20, double in_21, double in_22, double in_23, double in_24, double in_25, double in_26, double in_27, double in_28, double in_29, double in_30, double in_31, double in_32, double in_33, double in_34, double in_35, double in_36, double in_37, double in_38, double in_39, double in_40, double in_41, double in_42, double in_43, double in_44, double in_45, double in_46, double in_47, double in_48, double in_49, double in_50, double in_51, double in_52, double in_53, double in_54, double in_55, double in_56, double in_57, double in_58, double in_59, double in_60, double in_61, double in_62, double in_63, double in_64, double in_65, double in_66, double in_67, double in_68, double in_69, double in_70, double in_71, double in_72, double in_73, double in_74, double in_75, double in_76, double in_77, double in_78, double in_79, double in_80, double* out_0, double* out_1, double* out_2, double* out_3, double* out_4, double* out_5, double* out_6, double* out_7, double* out_8, double* out_9, double* out_10, double* out_11, double* out_12, double* out_13, double* out_14, double* out_15, double* out_16, double* out_17, double* out_18, double* out_19, double* out_20, double* out_21, double* out_22, double* out_23, double* out_24, double* out_25, double* out_26, double* out_27, double* out_28, double* out_29, double* out_30, double* out_31, double* out_32, double* out_33, double* out_34, double* out_35, double* out_36, double* out_37, double* out_38, double* out_39, double* out_40, double* out_41, double* out_42, double* out_43, double* out_44, double* out_45, double* out_46, double* out_47, double* out_48, double* out_49, double* out_50, double* out_51, double* out_52, double* out_53, double* out_54, double* out_55, double* out_56, double* out_57, double* out_58, double* out_59, double* out_60, double* out_61, double* out_62, double* out_63, double* out_64, double* out_65, double* out_66, double* out_67, double* out_68, double* out_69, double* out_70, double* out_71, double* out_72, double* out_73, double* out_74, double* out_75, double* out_76, double* out_77, double* out_78, double* out_79, double* out_80)
    {
        Eigen::Matrix<double, 9, 9> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(0, 4) = in_4; F(0, 5) = in_5; F(0, 6) = in_6; F(0, 7) = in_7; F(0, 8) = in_8; F(1, 0) = in_9; F(1, 1) = in_10; F(1, 2) = in_11; F(1, 3) = in_12; F(1, 4) = in_13; F(1, 5) = in_14; F(1, 6) = in_15; F(1, 7) = in_16; F(1, 8) = in_17; F(2, 0) = in_18; F(2, 1) = in_19; F(2, 2) = in_20; F(2, 3) = in_21; F(2, 4) = in_22; F(2, 5) = in_23; F(2, 6) = in_24; F(2, 7) = in_25; F(2, 8) = in_26; F(3, 0) = in_27; F(3, 1) = in_28; F(3, 2) = in_29; F(3, 3) = in_30; F(3, 4) = in_31; F(3, 5) = in_32; F(3, 6) = in_33; F(3, 7) = in_34; F(3, 8) = in_35; F(4, 0) = in_36; F(4, 1) = in_37; F(4, 2) = in_38; F(4, 3) = in_39; F(4, 4) = in_40; F(4, 5) = in_41; F(4, 6) = in_42; F(4, 7) = in_43; F(4, 8) = in_44; F(5, 0) = in_45; F(5, 1) = in_46; F(5, 2) = in_47; F(5, 3) = in_48; F(5, 4) = in_49; F(5, 5) = in_50; F(5, 6) = in_51; F(5, 7) = in_52; F(5, 8) = in_53; F(6, 0) = in_54; F(6, 1) = in_55; F(6, 2) = in_56; F(6, 3) = in_57; F(6, 4) = in_58; F(6, 5) = in_59; F(6, 6) = in_60; F(6, 7) = in_61; F(6, 8) = in_62; F(7, 0) = in_63; F(7, 1) = in_64; F(7, 2) = in_65; F(7, 3) = in_66; F(7, 4) = in_67; F(7, 5) = in_68; F(7, 6) = in_69; F(7, 7) = in_70; F(7, 8) = in_71; F(8, 0) = in_72; F(8, 1) = in_73; F(8, 2) = in_74; F(8, 3) = in_75; F(8, 4) = in_76; F(8, 5) = in_77; F(8, 6) = in_78; F(8, 7) = in_79; F(8, 8) = in_80;
        JGSL::makePD(F);
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(0, 4); out_5[0] = F(0, 5); out_6[0] = F(0, 6); out_7[0] = F(0, 7); out_8[0] = F(0, 8); out_9[0] = F(1, 0); out_10[0] = F(1, 1); out_11[0] = F(1, 2); out_12[0] = F(1, 3); out_13[0] = F(1, 4); out_14[0] = F(1, 5); out_15[0] = F(1, 6); out_16[0] = F(1, 7); out_17[0] = F(1, 8); out_18[0] = F(2, 0); out_19[0] = F(2, 1); out_20[0] = F(2, 2); out_21[0] = F(2, 3); out_22[0] = F(2, 4); out_23[0] = F(2, 5); out_24[0] = F(2, 6); out_25[0] = F(2, 7); out_26[0] = F(2, 8); out_27[0] = F(3, 0); out_28[0] = F(3, 1); out_29[0] = F(3, 2); out_30[0] = F(3, 3); out_31[0] = F(3, 4); out_32[0] = F(3, 5); out_33[0] = F(3, 6); out_34[0] = F(3, 7); out_35[0] = F(3, 8); out_36[0] = F(4, 0); out_37[0] = F(4, 1); out_38[0] = F(4, 2); out_39[0] = F(4, 3); out_40[0] = F(4, 4); out_41[0] = F(4, 5); out_42[0] = F(4, 6); out_43[0] = F(4, 7); out_44[0] = F(4, 8); out_45[0] = F(5, 0); out_46[0] = F(5, 1); out_47[0] = F(5, 2); out_48[0] = F(5, 3); out_49[0] = F(5, 4); out_50[0] = F(5, 5); out_51[0] = F(5, 6); out_52[0] = F(5, 7); out_53[0] = F(5, 8); out_54[0] = F(6, 0); out_55[0] = F(6, 1); out_56[0] = F(6, 2); out_57[0] = F(6, 3); out_58[0] = F(6, 4); out_59[0] = F(6, 5); out_60[0] = F(6, 6); out_61[0] = F(6, 7); out_62[0] = F(6, 8); out_63[0] = F(7, 0); out_64[0] = F(7, 1); out_65[0] = F(7, 2); out_66[0] = F(7, 3); out_67[0] = F(7, 4); out_68[0] = F(7, 5); out_69[0] = F(7, 6); out_70[0] = F(7, 7); out_71[0] = F(7, 8); out_72[0] = F(8, 0); out_73[0] = F(8, 1); out_74[0] = F(8, 2); out_75[0] = F(8, 3); out_76[0] = F(8, 4); out_77[0] = F(8, 5); out_78[0] = F(8, 6); out_79[0] = F(8, 7); out_80[0] = F(8, 8);
    }

    void inverse_6(double in_0, double in_1, double in_2, double in_3, double in_4, double in_5, double in_6, double in_7, double in_8, double in_9, double in_10, double in_11, double in_12, double in_13, double in_14, double in_15, double in_16, double in_17, double in_18, double in_19, double in_20, double in_21, double in_22, double in_23, double in_24, double in_25, double in_26, double in_27, double in_28, double in_29, double in_30, double in_31, double in_32, double in_33, double in_34, double in_35, double* out_0, double* out_1, double* out_2, double* out_3, double* out_4, double* out_5, double* out_6, double* out_7, double* out_8, double* out_9, double* out_10, double* out_11, double* out_12, double* out_13, double* out_14, double* out_15, double* out_16, double* out_17, double* out_18, double* out_19, double* out_20, double* out_21, double* out_22, double* out_23, double* out_24, double* out_25, double* out_26, double* out_27, double* out_28, double* out_29, double* out_30, double* out_31, double* out_32, double* out_33, double* out_34, double* out_35)
    {
        Eigen::Matrix<double, 6, 6> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(0, 4) = in_4; F(0, 5) = in_5; F(1, 0) = in_6; F(1, 1) = in_7; F(1, 2) = in_8; F(1, 3) = in_9; F(1, 4) = in_10; F(1, 5) = in_11; F(2, 0) = in_12; F(2, 1) = in_13; F(2, 2) = in_14; F(2, 3) = in_15; F(2, 4) = in_16; F(2, 5) = in_17; F(3, 0) = in_18; F(3, 1) = in_19; F(3, 2) = in_20; F(3, 3) = in_21; F(3, 4) = in_22; F(3, 5) = in_23; F(4, 0) = in_24; F(4, 1) = in_25; F(4, 2) = in_26; F(4, 3) = in_27; F(4, 4) = in_28; F(4, 5) = in_29; F(5, 0) = in_30; F(5, 1) = in_31; F(5, 2) = in_32; F(5, 3) = in_33; F(5, 4) = in_34; F(5, 5) = in_35;
        Eigen::Matrix<double, 6, 6> tmp = F;
        F = tmp.inverse();
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(0, 4); out_5[0] = F(0, 5); out_6[0] = F(1, 0); out_7[0] = F(1, 1); out_8[0] = F(1, 2); out_9[0] = F(1, 3); out_10[0] = F(1, 4); out_11[0] = F(1, 5); out_12[0] = F(2, 0); out_13[0] = F(2, 1); out_14[0] = F(2, 2); out_15[0] = F(2, 3); out_16[0] = F(2, 4); out_17[0] = F(2, 5); out_18[0] = F(3, 0); out_19[0] = F(3, 1); out_20[0] = F(3, 2); out_21[0] = F(3, 3); out_22[0] = F(3, 4); out_23[0] = F(3, 5); out_24[0] = F(4, 0); out_25[0] = F(4, 1); out_26[0] = F(4, 2); out_27[0] = F(4, 3); out_28[0] = F(4, 4); out_29[0] = F(4, 5); out_30[0] = F(5, 0); out_31[0] = F(5, 1); out_32[0] = F(5, 2); out_33[0] = F(5, 3); out_34[0] = F(5, 4); out_35[0] = F(5, 5);
    }

    void inverse_9(double in_0, double in_1, double in_2, double in_3, double in_4, double in_5, double in_6, double in_7, double in_8, double in_9, double in_10, double in_11, double in_12, double in_13, double in_14, double in_15, double in_16, double in_17, double in_18, double in_19, double in_20, double in_21, double in_22, double in_23, double in_24, double in_25, double in_26, double in_27, double in_28, double in_29, double in_30, double in_31, double in_32, double in_33, double in_34, double in_35, double in_36, double in_37, double in_38, double in_39, double in_40, double in_41, double in_42, double in_43, double in_44, double in_45, double in_46, double in_47, double in_48, double in_49, double in_50, double in_51, double in_52, double in_53, double in_54, double in_55, double in_56, double in_57, double in_58, double in_59, double in_60, double in_61, double in_62, double in_63, double in_64, double in_65, double in_66, double in_67, double in_68, double in_69, double in_70, double in_71, double in_72, double in_73, double in_74, double in_75, double in_76, double in_77, double in_78, double in_79, double in_80, double* out_0, double* out_1, double* out_2, double* out_3, double* out_4, double* out_5, double* out_6, double* out_7, double* out_8, double* out_9, double* out_10, double* out_11, double* out_12, double* out_13, double* out_14, double* out_15, double* out_16, double* out_17, double* out_18, double* out_19, double* out_20, double* out_21, double* out_22, double* out_23, double* out_24, double* out_25, double* out_26, double* out_27, double* out_28, double* out_29, double* out_30, double* out_31, double* out_32, double* out_33, double* out_34, double* out_35, double* out_36, double* out_37, double* out_38, double* out_39, double* out_40, double* out_41, double* out_42, double* out_43, double* out_44, double* out_45, double* out_46, double* out_47, double* out_48, double* out_49, double* out_50, double* out_51, double* out_52, double* out_53, double* out_54, double* out_55, double* out_56, double* out_57, double* out_58, double* out_59, double* out_60, double* out_61, double* out_62, double* out_63, double* out_64, double* out_65, double* out_66, double* out_67, double* out_68, double* out_69, double* out_70, double* out_71, double* out_72, double* out_73, double* out_74, double* out_75, double* out_76, double* out_77, double* out_78, double* out_79, double* out_80)
    {
        Eigen::Matrix<double, 9, 9> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(0, 4) = in_4; F(0, 5) = in_5; F(0, 6) = in_6; F(0, 7) = in_7; F(0, 8) = in_8; F(1, 0) = in_9; F(1, 1) = in_10; F(1, 2) = in_11; F(1, 3) = in_12; F(1, 4) = in_13; F(1, 5) = in_14; F(1, 6) = in_15; F(1, 7) = in_16; F(1, 8) = in_17; F(2, 0) = in_18; F(2, 1) = in_19; F(2, 2) = in_20; F(2, 3) = in_21; F(2, 4) = in_22; F(2, 5) = in_23; F(2, 6) = in_24; F(2, 7) = in_25; F(2, 8) = in_26; F(3, 0) = in_27; F(3, 1) = in_28; F(3, 2) = in_29; F(3, 3) = in_30; F(3, 4) = in_31; F(3, 5) = in_32; F(3, 6) = in_33; F(3, 7) = in_34; F(3, 8) = in_35; F(4, 0) = in_36; F(4, 1) = in_37; F(4, 2) = in_38; F(4, 3) = in_39; F(4, 4) = in_40; F(4, 5) = in_41; F(4, 6) = in_42; F(4, 7) = in_43; F(4, 8) = in_44; F(5, 0) = in_45; F(5, 1) = in_46; F(5, 2) = in_47; F(5, 3) = in_48; F(5, 4) = in_49; F(5, 5) = in_50; F(5, 6) = in_51; F(5, 7) = in_52; F(5, 8) = in_53; F(6, 0) = in_54; F(6, 1) = in_55; F(6, 2) = in_56; F(6, 3) = in_57; F(6, 4) = in_58; F(6, 5) = in_59; F(6, 6) = in_60; F(6, 7) = in_61; F(6, 8) = in_62; F(7, 0) = in_63; F(7, 1) = in_64; F(7, 2) = in_65; F(7, 3) = in_66; F(7, 4) = in_67; F(7, 5) = in_68; F(7, 6) = in_69; F(7, 7) = in_70; F(7, 8) = in_71; F(8, 0) = in_72; F(8, 1) = in_73; F(8, 2) = in_74; F(8, 3) = in_75; F(8, 4) = in_76; F(8, 5) = in_77; F(8, 6) = in_78; F(8, 7) = in_79; F(8, 8) = in_80;
        Eigen::Matrix<double, 9, 9> tmp = F;
        F = tmp.inverse();
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(0, 4); out_5[0] = F(0, 5); out_6[0] = F(0, 6); out_7[0] = F(0, 7); out_8[0] = F(0, 8); out_9[0] = F(1, 0); out_10[0] = F(1, 1); out_11[0] = F(1, 2); out_12[0] = F(1, 3); out_13[0] = F(1, 4); out_14[0] = F(1, 5); out_15[0] = F(1, 6); out_16[0] = F(1, 7); out_17[0] = F(1, 8); out_18[0] = F(2, 0); out_19[0] = F(2, 1); out_20[0] = F(2, 2); out_21[0] = F(2, 3); out_22[0] = F(2, 4); out_23[0] = F(2, 5); out_24[0] = F(2, 6); out_25[0] = F(2, 7); out_26[0] = F(2, 8); out_27[0] = F(3, 0); out_28[0] = F(3, 1); out_29[0] = F(3, 2); out_30[0] = F(3, 3); out_31[0] = F(3, 4); out_32[0] = F(3, 5); out_33[0] = F(3, 6); out_34[0] = F(3, 7); out_35[0] = F(3, 8); out_36[0] = F(4, 0); out_37[0] = F(4, 1); out_38[0] = F(4, 2); out_39[0] = F(4, 3); out_40[0] = F(4, 4); out_41[0] = F(4, 5); out_42[0] = F(4, 6); out_43[0] = F(4, 7); out_44[0] = F(4, 8); out_45[0] = F(5, 0); out_46[0] = F(5, 1); out_47[0] = F(5, 2); out_48[0] = F(5, 3); out_49[0] = F(5, 4); out_50[0] = F(5, 5); out_51[0] = F(5, 6); out_52[0] = F(5, 7); out_53[0] = F(5, 8); out_54[0] = F(6, 0); out_55[0] = F(6, 1); out_56[0] = F(6, 2); out_57[0] = F(6, 3); out_58[0] = F(6, 4); out_59[0] = F(6, 5); out_60[0] = F(6, 6); out_61[0] = F(6, 7); out_62[0] = F(6, 8); out_63[0] = F(7, 0); out_64[0] = F(7, 1); out_65[0] = F(7, 2); out_66[0] = F(7, 3); out_67[0] = F(7, 4); out_68[0] = F(7, 5); out_69[0] = F(7, 6); out_70[0] = F(7, 7); out_71[0] = F(7, 8); out_72[0] = F(8, 0); out_73[0] = F(8, 1); out_74[0] = F(8, 2); out_75[0] = F(8, 3); out_76[0] = F(8, 4); out_77[0] = F(8, 5); out_78[0] = F(8, 6); out_79[0] = F(8, 7); out_80[0] = F(8, 8);
    }

    void get_smallest_positive_real_quad_root(double a, double b, double c, double tol, double* ret)
    {
        // return negative value if no positive real root is found
        double t;

        if (abs(a) <= tol)
            t = -c / b;
        else {
            double desc = b * b - 4 * a * c;
            if (desc > 0) {
                t = (-b - sqrt(desc)) / (2 * a);
                if (t < 0)
                    t = (-b + sqrt(desc)) / (2 * a);
            }
            else // desv<0 ==> imag
                t = -1;
        }
        ret[0] = t;
    }

    void get_smallest_positive_real_cubic_root(double a, double b, double c, double d, double tol, double* ret)
    {
        // return negative value if no positive real root is found
        double t = -1;

        if (abs(a) <= tol)
            get_smallest_positive_real_quad_root(b, c, d, tol, &t);
        else {
            complex<double> i(0, 1);
            complex<double> delta0(b * b - 3 * a * c, 0);
            complex<double> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
            complex<double> C = pow((delta1 + sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
            if (std::abs(C) == 0.0) {
                // a corner case listed by wikipedia found by our collaborate from another project
                C = pow((delta1 - sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
            }

            complex<double> u2 = (-1.0 + sqrt(3.0) * i) / 2.0;
            complex<double> u3 = (-1.0 - sqrt(3.0) * i) / 2.0;

            complex<double> t1 = (b + C + delta0 / C) / (-3.0 * a);
            complex<double> t2 = (b + u2 * C + delta0 / (u2 * C)) / (-3.0 * a);
            complex<double> t3 = (b + u3 * C + delta0 / (u3 * C)) / (-3.0 * a);

            if ((abs(imag(t1)) < tol) && (real(t1) > 0))
                t = real(t1);
            if ((abs(imag(t2)) < tol) && (real(t2) > 0) && ((real(t2) < t) || (t < 0)))
                t = real(t2);
            if ((abs(imag(t3)) < tol) && (real(t3) > 0) && ((real(t3) < t) || (t < 0)))
                t = real(t3);
        }
        ret[0] = t;
    }

    void point_triangle_ccd(double p0, double p1, double p2,
                            double t00, double t01, double t02,
                            double t10, double t11, double t12,
                            double t20, double t21, double t22,
                            double dp0, double dp1, double dp2,
                            double dt00, double dt01, double dt02,
                            double dt10, double dt11, double dt12,
                            double dt20, double dt21, double dt22,
                            double eta, double dist2, double* ret)
    {
        Eigen::Matrix<double, 3, 1> p, t0, t1, t2, dp, dt0, dt1, dt2;
        p(0) = p0; p(1) = p1; p(2) = p2;
        t0(0) = t00; t0(1) = t01; t0(2) = t02;
        t1(0) = t10; t1(1) = t11; t1(2) = t12;
        t2(0) = t20; t2(1) = t21; t2(2) = t22;
        dp(0) = dp0; dp(1) = dp1; dp(2) = dp2;
        dt0(0) = dt00; dt0(1) = dt01; dt0(2) = dt02;
        dt1(0) = dt10; dt1(1) = dt11; dt1(2) = dt12;
        dt2(0) = dt20; dt2(1) = dt21; dt2(2) = dt22;
        double toc;
        if (CTCD::vertexFaceCTCD(p, t0, t1, t2,
                                 p + dp, t0 + dt0, t1 + dt1, t2 + dt2,
                                 eta * std::sqrt(dist2), toc))
        {
            if (toc < 1.0e-6) {
                puts("PT CCD tiny!");
                if (CTCD::vertexFaceCTCD(p, t0, t1, t2,
                                         p + dp, t0 + dt0, t1 + dt1, t2 + dt2,
                                         0, toc))
                {
                    toc *= (1.0 - eta);
                    ret[0] = (double)toc;
                }
                else {
                    ret[0] = 1.0;
                }
            }
            ret[0] = (double)toc;
        }
        else {
            ret[0] = 1.0;
        }
    }

    void edge_edge_ccd(double a00, double a01, double a02,
                       double a10, double a11, double a12,
                       double b00, double b01, double b02,
                       double b10, double b11, double b12,
                       double da00, double da01, double da02,
                       double da10, double da11, double da12,
                       double db00, double db01, double db02,
                       double db10, double db11, double db12,
                       double eta, double dist2, double* ret)
    {
        Eigen::Matrix<double, 3, 1> a0, a1, b0, b1, da0, da1, db0, db1;
        a0(0) = a00; a0(1) = a01; a0(2) = a02;
        a1(0) = a10; a1(1) = a11; a1(2) = a12;
        b0(0) = b00; b0(1) = b01; b0(2) = b02;
        b1(0) = b10; b1(1) = b11; b1(2) = b12;
        da0(0) = da00; da0(1) = da01; da0(2) = da02;
        da1(0) = da10; da1(1) = da11; da1(2) = da12;
        db0(0) = db00; db0(1) = db01; db0(2) = db02;
        db1(0) = db10; db1(1) = db11; db1(2) = db12;
        double toc;
        if (CTCD::edgeEdgeCTCD(a0, a1, b0, b1,
                               a0 + da0, a1 + da1, b0 + db0, b1 + db1,
                               eta * std::sqrt(dist2), toc))
        {
            if (toc < 1.0e-6) {
                puts("EE CCD tiny!");
                if (CTCD::edgeEdgeCTCD(a0, a1, b0, b1,
                                       a0 + da0, a1 + da1, b0 + db0, b1 + db1,
                                       0, toc))
                {
                    toc *= (1.0 - eta);
                    ret[0] = (double)toc;
                }
                else {
                    ret[0] = 1.0;
                }
            }
            ret[0] = (double)toc;
        }
        else {
            ret[0] = 1.0;
        }
    }


};