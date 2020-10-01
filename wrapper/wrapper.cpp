#include "SINGULAR_VALUE_DECOMPOSITION.h"
#include "VECTOR.h"
#include "UTILS.h"
#include "EVCTCD/CTCD.h"
#include <Eigen/Eigen>
#include <complex>
#include <cstdlib>
#define REAL double
using namespace std;

extern "C" {
    void svd(REAL F00, REAL F01, REAL F10, REAL F11,
             REAL* U00, REAL* U01, REAL* U10, REAL* U11,
             REAL* s00, REAL* s01, REAL* s10, REAL* s11,
             REAL* V00, REAL* V01, REAL* V10, REAL* V11)
    {
        JGSL::MATRIX<REAL, 2> F;
        F(0, 0) = F00; F(0, 1) = F01; F(1, 0) = F10; F(1, 1) = F11;
        JGSL::MATRIX<REAL, 2> U(1), V(1);
        JGSL::VECTOR<REAL, 2> sigma;
        JGSL::Singular_Value_Decomposition(F, U, sigma, V);
        U00[0] = U(0, 0),  U01[0] = U(0, 1), U10[0] = U(1, 0), U11[0] = U(1, 1);
        s00[0] = sigma(0), s01[0] = 0,      s10[0] = 0,       s11[0] = sigma(1);
        V00[0] = V(0, 0),  V01[0] = V(0, 1), V10[0] = V(1, 0), V11[0] = V(1, 1);
    }

    void project_pd_3(REAL in_0, REAL in_1, REAL in_2, REAL in_3, REAL in_4, REAL in_5, REAL in_6, REAL in_7, REAL in_8, REAL* out_0, REAL* out_1, REAL* out_2, REAL* out_3, REAL* out_4, REAL* out_5, REAL* out_6, REAL* out_7, REAL* out_8)
    {
        Eigen::Matrix<REAL, 3, 3> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(1, 0) = in_3; F(1, 1) = in_4; F(1, 2) = in_5; F(2, 0) = in_6; F(2, 1) = in_7; F(2, 2) = in_8;
        JGSL::makePD(F);
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(1, 0); out_4[0] = F(1, 1); out_5[0] = F(1, 2); out_6[0] = F(2, 0); out_7[0] = F(2, 1); out_8[0] = F(2, 2);
    }

    void project_pd_4(REAL in_0, REAL in_1, REAL in_2, REAL in_3, REAL in_4, REAL in_5, REAL in_6, REAL in_7, REAL in_8, REAL in_9, REAL in_10, REAL in_11, REAL in_12, REAL in_13, REAL in_14, REAL in_15, REAL* out_0, REAL* out_1, REAL* out_2, REAL* out_3, REAL* out_4, REAL* out_5, REAL* out_6, REAL* out_7, REAL* out_8, REAL* out_9, REAL* out_10, REAL* out_11, REAL* out_12, REAL* out_13, REAL* out_14, REAL* out_15)
    {
        Eigen::Matrix<REAL, 4, 4> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(1, 0) = in_4; F(1, 1) = in_5; F(1, 2) = in_6; F(1, 3) = in_7; F(2, 0) = in_8; F(2, 1) = in_9; F(2, 2) = in_10; F(2, 3) = in_11; F(3, 0) = in_12; F(3, 1) = in_13; F(3, 2) = in_14; F(3, 3) = in_15;
        JGSL::makePD(F);
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(1, 0); out_5[0] = F(1, 1); out_6[0] = F(1, 2); out_7[0] = F(1, 3); out_8[0] = F(2, 0); out_9[0] = F(2, 1); out_10[0] = F(2, 2); out_11[0] = F(2, 3); out_12[0] = F(3, 0); out_13[0] = F(3, 1); out_14[0] = F(3, 2); out_15[0] = F(3, 3);
    }

    void project_pd_6(REAL in_0, REAL in_1, REAL in_2, REAL in_3, REAL in_4, REAL in_5, REAL in_6, REAL in_7, REAL in_8, REAL in_9, REAL in_10, REAL in_11, REAL in_12, REAL in_13, REAL in_14, REAL in_15, REAL in_16, REAL in_17, REAL in_18, REAL in_19, REAL in_20, REAL in_21, REAL in_22, REAL in_23, REAL in_24, REAL in_25, REAL in_26, REAL in_27, REAL in_28, REAL in_29, REAL in_30, REAL in_31, REAL in_32, REAL in_33, REAL in_34, REAL in_35, REAL* out_0, REAL* out_1, REAL* out_2, REAL* out_3, REAL* out_4, REAL* out_5, REAL* out_6, REAL* out_7, REAL* out_8, REAL* out_9, REAL* out_10, REAL* out_11, REAL* out_12, REAL* out_13, REAL* out_14, REAL* out_15, REAL* out_16, REAL* out_17, REAL* out_18, REAL* out_19, REAL* out_20, REAL* out_21, REAL* out_22, REAL* out_23, REAL* out_24, REAL* out_25, REAL* out_26, REAL* out_27, REAL* out_28, REAL* out_29, REAL* out_30, REAL* out_31, REAL* out_32, REAL* out_33, REAL* out_34, REAL* out_35)
    {
        Eigen::Matrix<REAL, 6, 6> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(0, 4) = in_4; F(0, 5) = in_5; F(1, 0) = in_6; F(1, 1) = in_7; F(1, 2) = in_8; F(1, 3) = in_9; F(1, 4) = in_10; F(1, 5) = in_11; F(2, 0) = in_12; F(2, 1) = in_13; F(2, 2) = in_14; F(2, 3) = in_15; F(2, 4) = in_16; F(2, 5) = in_17; F(3, 0) = in_18; F(3, 1) = in_19; F(3, 2) = in_20; F(3, 3) = in_21; F(3, 4) = in_22; F(3, 5) = in_23; F(4, 0) = in_24; F(4, 1) = in_25; F(4, 2) = in_26; F(4, 3) = in_27; F(4, 4) = in_28; F(4, 5) = in_29; F(5, 0) = in_30; F(5, 1) = in_31; F(5, 2) = in_32; F(5, 3) = in_33; F(5, 4) = in_34; F(5, 5) = in_35;
        JGSL::makePD(F);
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(0, 4); out_5[0] = F(0, 5); out_6[0] = F(1, 0); out_7[0] = F(1, 1); out_8[0] = F(1, 2); out_9[0] = F(1, 3); out_10[0] = F(1, 4); out_11[0] = F(1, 5); out_12[0] = F(2, 0); out_13[0] = F(2, 1); out_14[0] = F(2, 2); out_15[0] = F(2, 3); out_16[0] = F(2, 4); out_17[0] = F(2, 5); out_18[0] = F(3, 0); out_19[0] = F(3, 1); out_20[0] = F(3, 2); out_21[0] = F(3, 3); out_22[0] = F(3, 4); out_23[0] = F(3, 5); out_24[0] = F(4, 0); out_25[0] = F(4, 1); out_26[0] = F(4, 2); out_27[0] = F(4, 3); out_28[0] = F(4, 4); out_29[0] = F(4, 5); out_30[0] = F(5, 0); out_31[0] = F(5, 1); out_32[0] = F(5, 2); out_33[0] = F(5, 3); out_34[0] = F(5, 4); out_35[0] = F(5, 5);
    }

    void project_pd_9(REAL in_0, REAL in_1, REAL in_2, REAL in_3, REAL in_4, REAL in_5, REAL in_6, REAL in_7, REAL in_8, REAL in_9, REAL in_10, REAL in_11, REAL in_12, REAL in_13, REAL in_14, REAL in_15, REAL in_16, REAL in_17, REAL in_18, REAL in_19, REAL in_20, REAL in_21, REAL in_22, REAL in_23, REAL in_24, REAL in_25, REAL in_26, REAL in_27, REAL in_28, REAL in_29, REAL in_30, REAL in_31, REAL in_32, REAL in_33, REAL in_34, REAL in_35, REAL in_36, REAL in_37, REAL in_38, REAL in_39, REAL in_40, REAL in_41, REAL in_42, REAL in_43, REAL in_44, REAL in_45, REAL in_46, REAL in_47, REAL in_48, REAL in_49, REAL in_50, REAL in_51, REAL in_52, REAL in_53, REAL in_54, REAL in_55, REAL in_56, REAL in_57, REAL in_58, REAL in_59, REAL in_60, REAL in_61, REAL in_62, REAL in_63, REAL in_64, REAL in_65, REAL in_66, REAL in_67, REAL in_68, REAL in_69, REAL in_70, REAL in_71, REAL in_72, REAL in_73, REAL in_74, REAL in_75, REAL in_76, REAL in_77, REAL in_78, REAL in_79, REAL in_80, REAL* out_0, REAL* out_1, REAL* out_2, REAL* out_3, REAL* out_4, REAL* out_5, REAL* out_6, REAL* out_7, REAL* out_8, REAL* out_9, REAL* out_10, REAL* out_11, REAL* out_12, REAL* out_13, REAL* out_14, REAL* out_15, REAL* out_16, REAL* out_17, REAL* out_18, REAL* out_19, REAL* out_20, REAL* out_21, REAL* out_22, REAL* out_23, REAL* out_24, REAL* out_25, REAL* out_26, REAL* out_27, REAL* out_28, REAL* out_29, REAL* out_30, REAL* out_31, REAL* out_32, REAL* out_33, REAL* out_34, REAL* out_35, REAL* out_36, REAL* out_37, REAL* out_38, REAL* out_39, REAL* out_40, REAL* out_41, REAL* out_42, REAL* out_43, REAL* out_44, REAL* out_45, REAL* out_46, REAL* out_47, REAL* out_48, REAL* out_49, REAL* out_50, REAL* out_51, REAL* out_52, REAL* out_53, REAL* out_54, REAL* out_55, REAL* out_56, REAL* out_57, REAL* out_58, REAL* out_59, REAL* out_60, REAL* out_61, REAL* out_62, REAL* out_63, REAL* out_64, REAL* out_65, REAL* out_66, REAL* out_67, REAL* out_68, REAL* out_69, REAL* out_70, REAL* out_71, REAL* out_72, REAL* out_73, REAL* out_74, REAL* out_75, REAL* out_76, REAL* out_77, REAL* out_78, REAL* out_79, REAL* out_80)
    {
        Eigen::Matrix<REAL, 9, 9> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(0, 4) = in_4; F(0, 5) = in_5; F(0, 6) = in_6; F(0, 7) = in_7; F(0, 8) = in_8; F(1, 0) = in_9; F(1, 1) = in_10; F(1, 2) = in_11; F(1, 3) = in_12; F(1, 4) = in_13; F(1, 5) = in_14; F(1, 6) = in_15; F(1, 7) = in_16; F(1, 8) = in_17; F(2, 0) = in_18; F(2, 1) = in_19; F(2, 2) = in_20; F(2, 3) = in_21; F(2, 4) = in_22; F(2, 5) = in_23; F(2, 6) = in_24; F(2, 7) = in_25; F(2, 8) = in_26; F(3, 0) = in_27; F(3, 1) = in_28; F(3, 2) = in_29; F(3, 3) = in_30; F(3, 4) = in_31; F(3, 5) = in_32; F(3, 6) = in_33; F(3, 7) = in_34; F(3, 8) = in_35; F(4, 0) = in_36; F(4, 1) = in_37; F(4, 2) = in_38; F(4, 3) = in_39; F(4, 4) = in_40; F(4, 5) = in_41; F(4, 6) = in_42; F(4, 7) = in_43; F(4, 8) = in_44; F(5, 0) = in_45; F(5, 1) = in_46; F(5, 2) = in_47; F(5, 3) = in_48; F(5, 4) = in_49; F(5, 5) = in_50; F(5, 6) = in_51; F(5, 7) = in_52; F(5, 8) = in_53; F(6, 0) = in_54; F(6, 1) = in_55; F(6, 2) = in_56; F(6, 3) = in_57; F(6, 4) = in_58; F(6, 5) = in_59; F(6, 6) = in_60; F(6, 7) = in_61; F(6, 8) = in_62; F(7, 0) = in_63; F(7, 1) = in_64; F(7, 2) = in_65; F(7, 3) = in_66; F(7, 4) = in_67; F(7, 5) = in_68; F(7, 6) = in_69; F(7, 7) = in_70; F(7, 8) = in_71; F(8, 0) = in_72; F(8, 1) = in_73; F(8, 2) = in_74; F(8, 3) = in_75; F(8, 4) = in_76; F(8, 5) = in_77; F(8, 6) = in_78; F(8, 7) = in_79; F(8, 8) = in_80;
        JGSL::makePD(F);
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(0, 4); out_5[0] = F(0, 5); out_6[0] = F(0, 6); out_7[0] = F(0, 7); out_8[0] = F(0, 8); out_9[0] = F(1, 0); out_10[0] = F(1, 1); out_11[0] = F(1, 2); out_12[0] = F(1, 3); out_13[0] = F(1, 4); out_14[0] = F(1, 5); out_15[0] = F(1, 6); out_16[0] = F(1, 7); out_17[0] = F(1, 8); out_18[0] = F(2, 0); out_19[0] = F(2, 1); out_20[0] = F(2, 2); out_21[0] = F(2, 3); out_22[0] = F(2, 4); out_23[0] = F(2, 5); out_24[0] = F(2, 6); out_25[0] = F(2, 7); out_26[0] = F(2, 8); out_27[0] = F(3, 0); out_28[0] = F(3, 1); out_29[0] = F(3, 2); out_30[0] = F(3, 3); out_31[0] = F(3, 4); out_32[0] = F(3, 5); out_33[0] = F(3, 6); out_34[0] = F(3, 7); out_35[0] = F(3, 8); out_36[0] = F(4, 0); out_37[0] = F(4, 1); out_38[0] = F(4, 2); out_39[0] = F(4, 3); out_40[0] = F(4, 4); out_41[0] = F(4, 5); out_42[0] = F(4, 6); out_43[0] = F(4, 7); out_44[0] = F(4, 8); out_45[0] = F(5, 0); out_46[0] = F(5, 1); out_47[0] = F(5, 2); out_48[0] = F(5, 3); out_49[0] = F(5, 4); out_50[0] = F(5, 5); out_51[0] = F(5, 6); out_52[0] = F(5, 7); out_53[0] = F(5, 8); out_54[0] = F(6, 0); out_55[0] = F(6, 1); out_56[0] = F(6, 2); out_57[0] = F(6, 3); out_58[0] = F(6, 4); out_59[0] = F(6, 5); out_60[0] = F(6, 6); out_61[0] = F(6, 7); out_62[0] = F(6, 8); out_63[0] = F(7, 0); out_64[0] = F(7, 1); out_65[0] = F(7, 2); out_66[0] = F(7, 3); out_67[0] = F(7, 4); out_68[0] = F(7, 5); out_69[0] = F(7, 6); out_70[0] = F(7, 7); out_71[0] = F(7, 8); out_72[0] = F(8, 0); out_73[0] = F(8, 1); out_74[0] = F(8, 2); out_75[0] = F(8, 3); out_76[0] = F(8, 4); out_77[0] = F(8, 5); out_78[0] = F(8, 6); out_79[0] = F(8, 7); out_80[0] = F(8, 8);
    }

    void inverse_2(REAL in_0, REAL in_1, REAL in_2, REAL in_3, REAL* out_0, REAL* out_1, REAL* out_2, REAL* out_3)
    {
        Eigen::Matrix<REAL, 2, 2> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(1, 0) = in_2; F(1, 1) = in_3;
        Eigen::Matrix<REAL, 2, 2> tmp = F;
        F = tmp.inverse();
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(1, 0); out_3[0] = F(1, 1);
    }

    void inverse_4(REAL in_0, REAL in_1, REAL in_2, REAL in_3, REAL in_4, REAL in_5, REAL in_6, REAL in_7, REAL in_8, REAL in_9, REAL in_10, REAL in_11, REAL in_12, REAL in_13, REAL in_14, REAL in_15, REAL* out_0, REAL* out_1, REAL* out_2, REAL* out_3, REAL* out_4, REAL* out_5, REAL* out_6, REAL* out_7, REAL* out_8, REAL* out_9, REAL* out_10, REAL* out_11, REAL* out_12, REAL* out_13, REAL* out_14, REAL* out_15)
    {
        Eigen::Matrix<REAL, 4, 4> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(1, 0) = in_4; F(1, 1) = in_5; F(1, 2) = in_6; F(1, 3) = in_7; F(2, 0) = in_8; F(2, 1) = in_9; F(2, 2) = in_10; F(2, 3) = in_11; F(3, 0) = in_12; F(3, 1) = in_13; F(3, 2) = in_14; F(3, 3) = in_15;
        Eigen::Matrix<REAL, 4, 4> tmp = F;
        F = tmp.inverse();
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(1, 0); out_5[0] = F(1, 1); out_6[0] = F(1, 2); out_7[0] = F(1, 3); out_8[0] = F(2, 0); out_9[0] = F(2, 1); out_10[0] = F(2, 2); out_11[0] = F(2, 3); out_12[0] = F(3, 0); out_13[0] = F(3, 1); out_14[0] = F(3, 2); out_15[0] = F(3, 3);
    }

    void inverse_6(REAL in_0, REAL in_1, REAL in_2, REAL in_3, REAL in_4, REAL in_5, REAL in_6, REAL in_7, REAL in_8, REAL in_9, REAL in_10, REAL in_11, REAL in_12, REAL in_13, REAL in_14, REAL in_15, REAL in_16, REAL in_17, REAL in_18, REAL in_19, REAL in_20, REAL in_21, REAL in_22, REAL in_23, REAL in_24, REAL in_25, REAL in_26, REAL in_27, REAL in_28, REAL in_29, REAL in_30, REAL in_31, REAL in_32, REAL in_33, REAL in_34, REAL in_35, REAL* out_0, REAL* out_1, REAL* out_2, REAL* out_3, REAL* out_4, REAL* out_5, REAL* out_6, REAL* out_7, REAL* out_8, REAL* out_9, REAL* out_10, REAL* out_11, REAL* out_12, REAL* out_13, REAL* out_14, REAL* out_15, REAL* out_16, REAL* out_17, REAL* out_18, REAL* out_19, REAL* out_20, REAL* out_21, REAL* out_22, REAL* out_23, REAL* out_24, REAL* out_25, REAL* out_26, REAL* out_27, REAL* out_28, REAL* out_29, REAL* out_30, REAL* out_31, REAL* out_32, REAL* out_33, REAL* out_34, REAL* out_35)
    {
        Eigen::Matrix<REAL, 6, 6> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(0, 4) = in_4; F(0, 5) = in_5; F(1, 0) = in_6; F(1, 1) = in_7; F(1, 2) = in_8; F(1, 3) = in_9; F(1, 4) = in_10; F(1, 5) = in_11; F(2, 0) = in_12; F(2, 1) = in_13; F(2, 2) = in_14; F(2, 3) = in_15; F(2, 4) = in_16; F(2, 5) = in_17; F(3, 0) = in_18; F(3, 1) = in_19; F(3, 2) = in_20; F(3, 3) = in_21; F(3, 4) = in_22; F(3, 5) = in_23; F(4, 0) = in_24; F(4, 1) = in_25; F(4, 2) = in_26; F(4, 3) = in_27; F(4, 4) = in_28; F(4, 5) = in_29; F(5, 0) = in_30; F(5, 1) = in_31; F(5, 2) = in_32; F(5, 3) = in_33; F(5, 4) = in_34; F(5, 5) = in_35;
        Eigen::Matrix<REAL, 6, 6> tmp = F;
        F = tmp.inverse();
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(0, 4); out_5[0] = F(0, 5); out_6[0] = F(1, 0); out_7[0] = F(1, 1); out_8[0] = F(1, 2); out_9[0] = F(1, 3); out_10[0] = F(1, 4); out_11[0] = F(1, 5); out_12[0] = F(2, 0); out_13[0] = F(2, 1); out_14[0] = F(2, 2); out_15[0] = F(2, 3); out_16[0] = F(2, 4); out_17[0] = F(2, 5); out_18[0] = F(3, 0); out_19[0] = F(3, 1); out_20[0] = F(3, 2); out_21[0] = F(3, 3); out_22[0] = F(3, 4); out_23[0] = F(3, 5); out_24[0] = F(4, 0); out_25[0] = F(4, 1); out_26[0] = F(4, 2); out_27[0] = F(4, 3); out_28[0] = F(4, 4); out_29[0] = F(4, 5); out_30[0] = F(5, 0); out_31[0] = F(5, 1); out_32[0] = F(5, 2); out_33[0] = F(5, 3); out_34[0] = F(5, 4); out_35[0] = F(5, 5);
    }

    void inverse_9(REAL in_0, REAL in_1, REAL in_2, REAL in_3, REAL in_4, REAL in_5, REAL in_6, REAL in_7, REAL in_8, REAL in_9, REAL in_10, REAL in_11, REAL in_12, REAL in_13, REAL in_14, REAL in_15, REAL in_16, REAL in_17, REAL in_18, REAL in_19, REAL in_20, REAL in_21, REAL in_22, REAL in_23, REAL in_24, REAL in_25, REAL in_26, REAL in_27, REAL in_28, REAL in_29, REAL in_30, REAL in_31, REAL in_32, REAL in_33, REAL in_34, REAL in_35, REAL in_36, REAL in_37, REAL in_38, REAL in_39, REAL in_40, REAL in_41, REAL in_42, REAL in_43, REAL in_44, REAL in_45, REAL in_46, REAL in_47, REAL in_48, REAL in_49, REAL in_50, REAL in_51, REAL in_52, REAL in_53, REAL in_54, REAL in_55, REAL in_56, REAL in_57, REAL in_58, REAL in_59, REAL in_60, REAL in_61, REAL in_62, REAL in_63, REAL in_64, REAL in_65, REAL in_66, REAL in_67, REAL in_68, REAL in_69, REAL in_70, REAL in_71, REAL in_72, REAL in_73, REAL in_74, REAL in_75, REAL in_76, REAL in_77, REAL in_78, REAL in_79, REAL in_80, REAL* out_0, REAL* out_1, REAL* out_2, REAL* out_3, REAL* out_4, REAL* out_5, REAL* out_6, REAL* out_7, REAL* out_8, REAL* out_9, REAL* out_10, REAL* out_11, REAL* out_12, REAL* out_13, REAL* out_14, REAL* out_15, REAL* out_16, REAL* out_17, REAL* out_18, REAL* out_19, REAL* out_20, REAL* out_21, REAL* out_22, REAL* out_23, REAL* out_24, REAL* out_25, REAL* out_26, REAL* out_27, REAL* out_28, REAL* out_29, REAL* out_30, REAL* out_31, REAL* out_32, REAL* out_33, REAL* out_34, REAL* out_35, REAL* out_36, REAL* out_37, REAL* out_38, REAL* out_39, REAL* out_40, REAL* out_41, REAL* out_42, REAL* out_43, REAL* out_44, REAL* out_45, REAL* out_46, REAL* out_47, REAL* out_48, REAL* out_49, REAL* out_50, REAL* out_51, REAL* out_52, REAL* out_53, REAL* out_54, REAL* out_55, REAL* out_56, REAL* out_57, REAL* out_58, REAL* out_59, REAL* out_60, REAL* out_61, REAL* out_62, REAL* out_63, REAL* out_64, REAL* out_65, REAL* out_66, REAL* out_67, REAL* out_68, REAL* out_69, REAL* out_70, REAL* out_71, REAL* out_72, REAL* out_73, REAL* out_74, REAL* out_75, REAL* out_76, REAL* out_77, REAL* out_78, REAL* out_79, REAL* out_80)
    {
        Eigen::Matrix<REAL, 9, 9> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(0, 4) = in_4; F(0, 5) = in_5; F(0, 6) = in_6; F(0, 7) = in_7; F(0, 8) = in_8; F(1, 0) = in_9; F(1, 1) = in_10; F(1, 2) = in_11; F(1, 3) = in_12; F(1, 4) = in_13; F(1, 5) = in_14; F(1, 6) = in_15; F(1, 7) = in_16; F(1, 8) = in_17; F(2, 0) = in_18; F(2, 1) = in_19; F(2, 2) = in_20; F(2, 3) = in_21; F(2, 4) = in_22; F(2, 5) = in_23; F(2, 6) = in_24; F(2, 7) = in_25; F(2, 8) = in_26; F(3, 0) = in_27; F(3, 1) = in_28; F(3, 2) = in_29; F(3, 3) = in_30; F(3, 4) = in_31; F(3, 5) = in_32; F(3, 6) = in_33; F(3, 7) = in_34; F(3, 8) = in_35; F(4, 0) = in_36; F(4, 1) = in_37; F(4, 2) = in_38; F(4, 3) = in_39; F(4, 4) = in_40; F(4, 5) = in_41; F(4, 6) = in_42; F(4, 7) = in_43; F(4, 8) = in_44; F(5, 0) = in_45; F(5, 1) = in_46; F(5, 2) = in_47; F(5, 3) = in_48; F(5, 4) = in_49; F(5, 5) = in_50; F(5, 6) = in_51; F(5, 7) = in_52; F(5, 8) = in_53; F(6, 0) = in_54; F(6, 1) = in_55; F(6, 2) = in_56; F(6, 3) = in_57; F(6, 4) = in_58; F(6, 5) = in_59; F(6, 6) = in_60; F(6, 7) = in_61; F(6, 8) = in_62; F(7, 0) = in_63; F(7, 1) = in_64; F(7, 2) = in_65; F(7, 3) = in_66; F(7, 4) = in_67; F(7, 5) = in_68; F(7, 6) = in_69; F(7, 7) = in_70; F(7, 8) = in_71; F(8, 0) = in_72; F(8, 1) = in_73; F(8, 2) = in_74; F(8, 3) = in_75; F(8, 4) = in_76; F(8, 5) = in_77; F(8, 6) = in_78; F(8, 7) = in_79; F(8, 8) = in_80;
        Eigen::Matrix<REAL, 9, 9> tmp = F;
        F = tmp.inverse();
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(0, 4); out_5[0] = F(0, 5); out_6[0] = F(0, 6); out_7[0] = F(0, 7); out_8[0] = F(0, 8); out_9[0] = F(1, 0); out_10[0] = F(1, 1); out_11[0] = F(1, 2); out_12[0] = F(1, 3); out_13[0] = F(1, 4); out_14[0] = F(1, 5); out_15[0] = F(1, 6); out_16[0] = F(1, 7); out_17[0] = F(1, 8); out_18[0] = F(2, 0); out_19[0] = F(2, 1); out_20[0] = F(2, 2); out_21[0] = F(2, 3); out_22[0] = F(2, 4); out_23[0] = F(2, 5); out_24[0] = F(2, 6); out_25[0] = F(2, 7); out_26[0] = F(2, 8); out_27[0] = F(3, 0); out_28[0] = F(3, 1); out_29[0] = F(3, 2); out_30[0] = F(3, 3); out_31[0] = F(3, 4); out_32[0] = F(3, 5); out_33[0] = F(3, 6); out_34[0] = F(3, 7); out_35[0] = F(3, 8); out_36[0] = F(4, 0); out_37[0] = F(4, 1); out_38[0] = F(4, 2); out_39[0] = F(4, 3); out_40[0] = F(4, 4); out_41[0] = F(4, 5); out_42[0] = F(4, 6); out_43[0] = F(4, 7); out_44[0] = F(4, 8); out_45[0] = F(5, 0); out_46[0] = F(5, 1); out_47[0] = F(5, 2); out_48[0] = F(5, 3); out_49[0] = F(5, 4); out_50[0] = F(5, 5); out_51[0] = F(5, 6); out_52[0] = F(5, 7); out_53[0] = F(5, 8); out_54[0] = F(6, 0); out_55[0] = F(6, 1); out_56[0] = F(6, 2); out_57[0] = F(6, 3); out_58[0] = F(6, 4); out_59[0] = F(6, 5); out_60[0] = F(6, 6); out_61[0] = F(6, 7); out_62[0] = F(6, 8); out_63[0] = F(7, 0); out_64[0] = F(7, 1); out_65[0] = F(7, 2); out_66[0] = F(7, 3); out_67[0] = F(7, 4); out_68[0] = F(7, 5); out_69[0] = F(7, 6); out_70[0] = F(7, 7); out_71[0] = F(7, 8); out_72[0] = F(8, 0); out_73[0] = F(8, 1); out_74[0] = F(8, 2); out_75[0] = F(8, 3); out_76[0] = F(8, 4); out_77[0] = F(8, 5); out_78[0] = F(8, 6); out_79[0] = F(8, 7); out_80[0] = F(8, 8);
    }

    void get_smallest_positive_real_quad_root(REAL a, REAL b, REAL c, REAL tol, REAL* ret)
    {
        // return negative value if no positive real root is found
        REAL t;

        if (abs(a) <= tol)
            t = -c / b;
        else {
            REAL desc = b * b - 4 * a * c;
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

    void get_smallest_positive_real_cubic_root(REAL a, REAL b, REAL c, REAL d, REAL tol, REAL* ret)
    {
        // return negative value if no positive real root is found
        REAL t = -1;

        if (abs(a) <= tol)
            get_smallest_positive_real_quad_root(b, c, d, tol, &t);
        else {
            complex<REAL> i(0, 1);
            complex<REAL> delta0(b * b - 3 * a * c, 0);
            complex<REAL> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
            complex<REAL> C = pow((delta1 + sqrt(delta1 * delta1 - REAL(4.0) * delta0 * delta0 * delta0)) / REAL(2.0), REAL(1.0 / 3.0));
            if (std::abs(C) == 0.0) {
                // a corner case listed by wikipedia found by our collaborate from another project
                C = pow((delta1 - sqrt(delta1 * delta1 - REAL(4.0) * delta0 * delta0 * delta0)) / REAL(2.0), REAL(1.0 / 3.0));
            }

            complex<REAL> u2 = (REAL(-1.0) + REAL(sqrt(3.0)) * i) / REAL(2.0);
            complex<REAL> u3 = (REAL(-1.0) - REAL(sqrt(3.0)) * i) / REAL(2.0);

            complex<REAL> t1 = (b + C + delta0 / C) / REAL(-3.0 * a);
            complex<REAL> t2 = (b + u2 * C + delta0 / (u2 * C)) / REAL(-3.0 * a);
            complex<REAL> t3 = (b + u3 * C + delta0 / (u3 * C)) / REAL(-3.0 * a);

            if ((abs(imag(t1)) < tol) && (real(t1) > 0))
                t = real(t1);
            if ((abs(imag(t2)) < tol) && (real(t2) > 0) && ((real(t2) < t) || (t < 0)))
                t = real(t2);
            if ((abs(imag(t3)) < tol) && (real(t3) > 0) && ((real(t3) < t) || (t < 0)))
                t = real(t3);
        }
        ret[0] = t;
    }

    void point_triangle_ccd(REAL p0, REAL p1, REAL p2,
                            REAL t00, REAL t01, REAL t02,
                            REAL t10, REAL t11, REAL t12,
                            REAL t20, REAL t21, REAL t22,
                            REAL dp0, REAL dp1, REAL dp2,
                            REAL dt00, REAL dt01, REAL dt02,
                            REAL dt10, REAL dt11, REAL dt12,
                            REAL dt20, REAL dt21, REAL dt22,
                            REAL eta, REAL dist2, REAL* ret)
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
                    ret[0] = (REAL)toc;
                }
                else {
                    ret[0] = 1.0;
                }
            }
            ret[0] = (REAL)toc;
        }
        else {
            ret[0] = 1.0;
        }
    }

    void edge_edge_ccd(REAL a00, REAL a01, REAL a02,
                       REAL a10, REAL a11, REAL a12,
                       REAL b00, REAL b01, REAL b02,
                       REAL b10, REAL b11, REAL b12,
                       REAL da00, REAL da01, REAL da02,
                       REAL da10, REAL da11, REAL da12,
                       REAL db00, REAL db01, REAL db02,
                       REAL db10, REAL db11, REAL db12,
                       REAL eta, REAL dist2, REAL* ret)
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
                    ret[0] = (REAL)toc;
                }
                else {
                    ret[0] = 1.0;
                }
            }
            ret[0] = (REAL)toc;
        }
        else {
            ret[0] = 1.0;
        }
    }


};