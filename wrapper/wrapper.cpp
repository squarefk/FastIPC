#include "SINGULAR_VALUE_DECOMPOSITION.h"
#include "VECTOR.h"
#include "UTILS.h"
#include "EVCTCD/CTCD.h"
#include <Eigen/Eigen>

extern "C" {
    void svd(float F00, float F01, float F10, float F11,
             float* U00, float* U01, float* U10, float* U11,
             float* s00, float* s01, float* s10, float* s11,
             float* V00, float* V01, float* V10, float* V11)
    {
        JGSL::MATRIX<float, 2> F;
        F(0, 0) = F00; F(0, 1) = F01; F(1, 0) = F10; F(1, 1) = F11;
        JGSL::MATRIX<float, 2> U(1), V(1);
        JGSL::VECTOR<float, 2> sigma;
        JGSL::Singular_Value_Decomposition(F, U, sigma, V);
        U00[0] = U(0, 0),  U01[0] = U(0, 1), U10[0] = U(1, 0), U11[0] = U(1, 1);
        s00[0] = sigma(0), s01[0] = 0,      s10[0] = 0,       s11[0] = sigma(1);
        V00[0] = V(0, 0),  V01[0] = V(0, 1), V10[0] = V(1, 0), V11[0] = V(1, 1);
    }

    void project_pd_3(float in_0, float in_1, float in_2, float in_3, float in_4, float in_5, float in_6, float in_7, float in_8, float* out_0, float* out_1, float* out_2, float* out_3, float* out_4, float* out_5, float* out_6, float* out_7, float* out_8)
    {
        Eigen::Matrix<float, 3, 3> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(1, 0) = in_3; F(1, 1) = in_4; F(1, 2) = in_5; F(2, 0) = in_6; F(2, 1) = in_7; F(2, 2) = in_8;
        JGSL::makePD(F);
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(1, 0); out_4[0] = F(1, 1); out_5[0] = F(1, 2); out_6[0] = F(2, 0); out_7[0] = F(2, 1); out_8[0] = F(2, 2);
    }

    void project_pd_6(float in_0, float in_1, float in_2, float in_3, float in_4, float in_5, float in_6, float in_7, float in_8, float in_9, float in_10, float in_11, float in_12, float in_13, float in_14, float in_15, float in_16, float in_17, float in_18, float in_19, float in_20, float in_21, float in_22, float in_23, float in_24, float in_25, float in_26, float in_27, float in_28, float in_29, float in_30, float in_31, float in_32, float in_33, float in_34, float in_35, float* out_0, float* out_1, float* out_2, float* out_3, float* out_4, float* out_5, float* out_6, float* out_7, float* out_8, float* out_9, float* out_10, float* out_11, float* out_12, float* out_13, float* out_14, float* out_15, float* out_16, float* out_17, float* out_18, float* out_19, float* out_20, float* out_21, float* out_22, float* out_23, float* out_24, float* out_25, float* out_26, float* out_27, float* out_28, float* out_29, float* out_30, float* out_31, float* out_32, float* out_33, float* out_34, float* out_35)
    {
        Eigen::Matrix<float, 6, 6> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(0, 4) = in_4; F(0, 5) = in_5; F(1, 0) = in_6; F(1, 1) = in_7; F(1, 2) = in_8; F(1, 3) = in_9; F(1, 4) = in_10; F(1, 5) = in_11; F(2, 0) = in_12; F(2, 1) = in_13; F(2, 2) = in_14; F(2, 3) = in_15; F(2, 4) = in_16; F(2, 5) = in_17; F(3, 0) = in_18; F(3, 1) = in_19; F(3, 2) = in_20; F(3, 3) = in_21; F(3, 4) = in_22; F(3, 5) = in_23; F(4, 0) = in_24; F(4, 1) = in_25; F(4, 2) = in_26; F(4, 3) = in_27; F(4, 4) = in_28; F(4, 5) = in_29; F(5, 0) = in_30; F(5, 1) = in_31; F(5, 2) = in_32; F(5, 3) = in_33; F(5, 4) = in_34; F(5, 5) = in_35;
        JGSL::makePD(F);
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(0, 4); out_5[0] = F(0, 5); out_6[0] = F(1, 0); out_7[0] = F(1, 1); out_8[0] = F(1, 2); out_9[0] = F(1, 3); out_10[0] = F(1, 4); out_11[0] = F(1, 5); out_12[0] = F(2, 0); out_13[0] = F(2, 1); out_14[0] = F(2, 2); out_15[0] = F(2, 3); out_16[0] = F(2, 4); out_17[0] = F(2, 5); out_18[0] = F(3, 0); out_19[0] = F(3, 1); out_20[0] = F(3, 2); out_21[0] = F(3, 3); out_22[0] = F(3, 4); out_23[0] = F(3, 5); out_24[0] = F(4, 0); out_25[0] = F(4, 1); out_26[0] = F(4, 2); out_27[0] = F(4, 3); out_28[0] = F(4, 4); out_29[0] = F(4, 5); out_30[0] = F(5, 0); out_31[0] = F(5, 1); out_32[0] = F(5, 2); out_33[0] = F(5, 3); out_34[0] = F(5, 4); out_35[0] = F(5, 5);
    }

    void project_pd_9(float in_0, float in_1, float in_2, float in_3, float in_4, float in_5, float in_6, float in_7, float in_8, float in_9, float in_10, float in_11, float in_12, float in_13, float in_14, float in_15, float in_16, float in_17, float in_18, float in_19, float in_20, float in_21, float in_22, float in_23, float in_24, float in_25, float in_26, float in_27, float in_28, float in_29, float in_30, float in_31, float in_32, float in_33, float in_34, float in_35, float in_36, float in_37, float in_38, float in_39, float in_40, float in_41, float in_42, float in_43, float in_44, float in_45, float in_46, float in_47, float in_48, float in_49, float in_50, float in_51, float in_52, float in_53, float in_54, float in_55, float in_56, float in_57, float in_58, float in_59, float in_60, float in_61, float in_62, float in_63, float in_64, float in_65, float in_66, float in_67, float in_68, float in_69, float in_70, float in_71, float in_72, float in_73, float in_74, float in_75, float in_76, float in_77, float in_78, float in_79, float in_80, float* out_0, float* out_1, float* out_2, float* out_3, float* out_4, float* out_5, float* out_6, float* out_7, float* out_8, float* out_9, float* out_10, float* out_11, float* out_12, float* out_13, float* out_14, float* out_15, float* out_16, float* out_17, float* out_18, float* out_19, float* out_20, float* out_21, float* out_22, float* out_23, float* out_24, float* out_25, float* out_26, float* out_27, float* out_28, float* out_29, float* out_30, float* out_31, float* out_32, float* out_33, float* out_34, float* out_35, float* out_36, float* out_37, float* out_38, float* out_39, float* out_40, float* out_41, float* out_42, float* out_43, float* out_44, float* out_45, float* out_46, float* out_47, float* out_48, float* out_49, float* out_50, float* out_51, float* out_52, float* out_53, float* out_54, float* out_55, float* out_56, float* out_57, float* out_58, float* out_59, float* out_60, float* out_61, float* out_62, float* out_63, float* out_64, float* out_65, float* out_66, float* out_67, float* out_68, float* out_69, float* out_70, float* out_71, float* out_72, float* out_73, float* out_74, float* out_75, float* out_76, float* out_77, float* out_78, float* out_79, float* out_80)
    {
        Eigen::Matrix<float, 9, 9> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(0, 4) = in_4; F(0, 5) = in_5; F(0, 6) = in_6; F(0, 7) = in_7; F(0, 8) = in_8; F(1, 0) = in_9; F(1, 1) = in_10; F(1, 2) = in_11; F(1, 3) = in_12; F(1, 4) = in_13; F(1, 5) = in_14; F(1, 6) = in_15; F(1, 7) = in_16; F(1, 8) = in_17; F(2, 0) = in_18; F(2, 1) = in_19; F(2, 2) = in_20; F(2, 3) = in_21; F(2, 4) = in_22; F(2, 5) = in_23; F(2, 6) = in_24; F(2, 7) = in_25; F(2, 8) = in_26; F(3, 0) = in_27; F(3, 1) = in_28; F(3, 2) = in_29; F(3, 3) = in_30; F(3, 4) = in_31; F(3, 5) = in_32; F(3, 6) = in_33; F(3, 7) = in_34; F(3, 8) = in_35; F(4, 0) = in_36; F(4, 1) = in_37; F(4, 2) = in_38; F(4, 3) = in_39; F(4, 4) = in_40; F(4, 5) = in_41; F(4, 6) = in_42; F(4, 7) = in_43; F(4, 8) = in_44; F(5, 0) = in_45; F(5, 1) = in_46; F(5, 2) = in_47; F(5, 3) = in_48; F(5, 4) = in_49; F(5, 5) = in_50; F(5, 6) = in_51; F(5, 7) = in_52; F(5, 8) = in_53; F(6, 0) = in_54; F(6, 1) = in_55; F(6, 2) = in_56; F(6, 3) = in_57; F(6, 4) = in_58; F(6, 5) = in_59; F(6, 6) = in_60; F(6, 7) = in_61; F(6, 8) = in_62; F(7, 0) = in_63; F(7, 1) = in_64; F(7, 2) = in_65; F(7, 3) = in_66; F(7, 4) = in_67; F(7, 5) = in_68; F(7, 6) = in_69; F(7, 7) = in_70; F(7, 8) = in_71; F(8, 0) = in_72; F(8, 1) = in_73; F(8, 2) = in_74; F(8, 3) = in_75; F(8, 4) = in_76; F(8, 5) = in_77; F(8, 6) = in_78; F(8, 7) = in_79; F(8, 8) = in_80;
        JGSL::makePD(F);
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(0, 4); out_5[0] = F(0, 5); out_6[0] = F(0, 6); out_7[0] = F(0, 7); out_8[0] = F(0, 8); out_9[0] = F(1, 0); out_10[0] = F(1, 1); out_11[0] = F(1, 2); out_12[0] = F(1, 3); out_13[0] = F(1, 4); out_14[0] = F(1, 5); out_15[0] = F(1, 6); out_16[0] = F(1, 7); out_17[0] = F(1, 8); out_18[0] = F(2, 0); out_19[0] = F(2, 1); out_20[0] = F(2, 2); out_21[0] = F(2, 3); out_22[0] = F(2, 4); out_23[0] = F(2, 5); out_24[0] = F(2, 6); out_25[0] = F(2, 7); out_26[0] = F(2, 8); out_27[0] = F(3, 0); out_28[0] = F(3, 1); out_29[0] = F(3, 2); out_30[0] = F(3, 3); out_31[0] = F(3, 4); out_32[0] = F(3, 5); out_33[0] = F(3, 6); out_34[0] = F(3, 7); out_35[0] = F(3, 8); out_36[0] = F(4, 0); out_37[0] = F(4, 1); out_38[0] = F(4, 2); out_39[0] = F(4, 3); out_40[0] = F(4, 4); out_41[0] = F(4, 5); out_42[0] = F(4, 6); out_43[0] = F(4, 7); out_44[0] = F(4, 8); out_45[0] = F(5, 0); out_46[0] = F(5, 1); out_47[0] = F(5, 2); out_48[0] = F(5, 3); out_49[0] = F(5, 4); out_50[0] = F(5, 5); out_51[0] = F(5, 6); out_52[0] = F(5, 7); out_53[0] = F(5, 8); out_54[0] = F(6, 0); out_55[0] = F(6, 1); out_56[0] = F(6, 2); out_57[0] = F(6, 3); out_58[0] = F(6, 4); out_59[0] = F(6, 5); out_60[0] = F(6, 6); out_61[0] = F(6, 7); out_62[0] = F(6, 8); out_63[0] = F(7, 0); out_64[0] = F(7, 1); out_65[0] = F(7, 2); out_66[0] = F(7, 3); out_67[0] = F(7, 4); out_68[0] = F(7, 5); out_69[0] = F(7, 6); out_70[0] = F(7, 7); out_71[0] = F(7, 8); out_72[0] = F(8, 0); out_73[0] = F(8, 1); out_74[0] = F(8, 2); out_75[0] = F(8, 3); out_76[0] = F(8, 4); out_77[0] = F(8, 5); out_78[0] = F(8, 6); out_79[0] = F(8, 7); out_80[0] = F(8, 8);
    }

    void inverse_6(float in_0, float in_1, float in_2, float in_3, float in_4, float in_5, float in_6, float in_7, float in_8, float in_9, float in_10, float in_11, float in_12, float in_13, float in_14, float in_15, float in_16, float in_17, float in_18, float in_19, float in_20, float in_21, float in_22, float in_23, float in_24, float in_25, float in_26, float in_27, float in_28, float in_29, float in_30, float in_31, float in_32, float in_33, float in_34, float in_35, float* out_0, float* out_1, float* out_2, float* out_3, float* out_4, float* out_5, float* out_6, float* out_7, float* out_8, float* out_9, float* out_10, float* out_11, float* out_12, float* out_13, float* out_14, float* out_15, float* out_16, float* out_17, float* out_18, float* out_19, float* out_20, float* out_21, float* out_22, float* out_23, float* out_24, float* out_25, float* out_26, float* out_27, float* out_28, float* out_29, float* out_30, float* out_31, float* out_32, float* out_33, float* out_34, float* out_35)
    {
        Eigen::Matrix<float, 6, 6> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(0, 4) = in_4; F(0, 5) = in_5; F(1, 0) = in_6; F(1, 1) = in_7; F(1, 2) = in_8; F(1, 3) = in_9; F(1, 4) = in_10; F(1, 5) = in_11; F(2, 0) = in_12; F(2, 1) = in_13; F(2, 2) = in_14; F(2, 3) = in_15; F(2, 4) = in_16; F(2, 5) = in_17; F(3, 0) = in_18; F(3, 1) = in_19; F(3, 2) = in_20; F(3, 3) = in_21; F(3, 4) = in_22; F(3, 5) = in_23; F(4, 0) = in_24; F(4, 1) = in_25; F(4, 2) = in_26; F(4, 3) = in_27; F(4, 4) = in_28; F(4, 5) = in_29; F(5, 0) = in_30; F(5, 1) = in_31; F(5, 2) = in_32; F(5, 3) = in_33; F(5, 4) = in_34; F(5, 5) = in_35;
        Eigen::Matrix<float, 6, 6> tmp = F;
        F = tmp.inverse();
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(0, 4); out_5[0] = F(0, 5); out_6[0] = F(1, 0); out_7[0] = F(1, 1); out_8[0] = F(1, 2); out_9[0] = F(1, 3); out_10[0] = F(1, 4); out_11[0] = F(1, 5); out_12[0] = F(2, 0); out_13[0] = F(2, 1); out_14[0] = F(2, 2); out_15[0] = F(2, 3); out_16[0] = F(2, 4); out_17[0] = F(2, 5); out_18[0] = F(3, 0); out_19[0] = F(3, 1); out_20[0] = F(3, 2); out_21[0] = F(3, 3); out_22[0] = F(3, 4); out_23[0] = F(3, 5); out_24[0] = F(4, 0); out_25[0] = F(4, 1); out_26[0] = F(4, 2); out_27[0] = F(4, 3); out_28[0] = F(4, 4); out_29[0] = F(4, 5); out_30[0] = F(5, 0); out_31[0] = F(5, 1); out_32[0] = F(5, 2); out_33[0] = F(5, 3); out_34[0] = F(5, 4); out_35[0] = F(5, 5);
    }

    void inverse_9(float in_0, float in_1, float in_2, float in_3, float in_4, float in_5, float in_6, float in_7, float in_8, float in_9, float in_10, float in_11, float in_12, float in_13, float in_14, float in_15, float in_16, float in_17, float in_18, float in_19, float in_20, float in_21, float in_22, float in_23, float in_24, float in_25, float in_26, float in_27, float in_28, float in_29, float in_30, float in_31, float in_32, float in_33, float in_34, float in_35, float in_36, float in_37, float in_38, float in_39, float in_40, float in_41, float in_42, float in_43, float in_44, float in_45, float in_46, float in_47, float in_48, float in_49, float in_50, float in_51, float in_52, float in_53, float in_54, float in_55, float in_56, float in_57, float in_58, float in_59, float in_60, float in_61, float in_62, float in_63, float in_64, float in_65, float in_66, float in_67, float in_68, float in_69, float in_70, float in_71, float in_72, float in_73, float in_74, float in_75, float in_76, float in_77, float in_78, float in_79, float in_80, float* out_0, float* out_1, float* out_2, float* out_3, float* out_4, float* out_5, float* out_6, float* out_7, float* out_8, float* out_9, float* out_10, float* out_11, float* out_12, float* out_13, float* out_14, float* out_15, float* out_16, float* out_17, float* out_18, float* out_19, float* out_20, float* out_21, float* out_22, float* out_23, float* out_24, float* out_25, float* out_26, float* out_27, float* out_28, float* out_29, float* out_30, float* out_31, float* out_32, float* out_33, float* out_34, float* out_35, float* out_36, float* out_37, float* out_38, float* out_39, float* out_40, float* out_41, float* out_42, float* out_43, float* out_44, float* out_45, float* out_46, float* out_47, float* out_48, float* out_49, float* out_50, float* out_51, float* out_52, float* out_53, float* out_54, float* out_55, float* out_56, float* out_57, float* out_58, float* out_59, float* out_60, float* out_61, float* out_62, float* out_63, float* out_64, float* out_65, float* out_66, float* out_67, float* out_68, float* out_69, float* out_70, float* out_71, float* out_72, float* out_73, float* out_74, float* out_75, float* out_76, float* out_77, float* out_78, float* out_79, float* out_80)
    {
        Eigen::Matrix<float, 9, 9> F;
        F(0, 0) = in_0; F(0, 1) = in_1; F(0, 2) = in_2; F(0, 3) = in_3; F(0, 4) = in_4; F(0, 5) = in_5; F(0, 6) = in_6; F(0, 7) = in_7; F(0, 8) = in_8; F(1, 0) = in_9; F(1, 1) = in_10; F(1, 2) = in_11; F(1, 3) = in_12; F(1, 4) = in_13; F(1, 5) = in_14; F(1, 6) = in_15; F(1, 7) = in_16; F(1, 8) = in_17; F(2, 0) = in_18; F(2, 1) = in_19; F(2, 2) = in_20; F(2, 3) = in_21; F(2, 4) = in_22; F(2, 5) = in_23; F(2, 6) = in_24; F(2, 7) = in_25; F(2, 8) = in_26; F(3, 0) = in_27; F(3, 1) = in_28; F(3, 2) = in_29; F(3, 3) = in_30; F(3, 4) = in_31; F(3, 5) = in_32; F(3, 6) = in_33; F(3, 7) = in_34; F(3, 8) = in_35; F(4, 0) = in_36; F(4, 1) = in_37; F(4, 2) = in_38; F(4, 3) = in_39; F(4, 4) = in_40; F(4, 5) = in_41; F(4, 6) = in_42; F(4, 7) = in_43; F(4, 8) = in_44; F(5, 0) = in_45; F(5, 1) = in_46; F(5, 2) = in_47; F(5, 3) = in_48; F(5, 4) = in_49; F(5, 5) = in_50; F(5, 6) = in_51; F(5, 7) = in_52; F(5, 8) = in_53; F(6, 0) = in_54; F(6, 1) = in_55; F(6, 2) = in_56; F(6, 3) = in_57; F(6, 4) = in_58; F(6, 5) = in_59; F(6, 6) = in_60; F(6, 7) = in_61; F(6, 8) = in_62; F(7, 0) = in_63; F(7, 1) = in_64; F(7, 2) = in_65; F(7, 3) = in_66; F(7, 4) = in_67; F(7, 5) = in_68; F(7, 6) = in_69; F(7, 7) = in_70; F(7, 8) = in_71; F(8, 0) = in_72; F(8, 1) = in_73; F(8, 2) = in_74; F(8, 3) = in_75; F(8, 4) = in_76; F(8, 5) = in_77; F(8, 6) = in_78; F(8, 7) = in_79; F(8, 8) = in_80;
        Eigen::Matrix<float, 9, 9> tmp = F;
        F = tmp.inverse();
        out_0[0] = F(0, 0); out_1[0] = F(0, 1); out_2[0] = F(0, 2); out_3[0] = F(0, 3); out_4[0] = F(0, 4); out_5[0] = F(0, 5); out_6[0] = F(0, 6); out_7[0] = F(0, 7); out_8[0] = F(0, 8); out_9[0] = F(1, 0); out_10[0] = F(1, 1); out_11[0] = F(1, 2); out_12[0] = F(1, 3); out_13[0] = F(1, 4); out_14[0] = F(1, 5); out_15[0] = F(1, 6); out_16[0] = F(1, 7); out_17[0] = F(1, 8); out_18[0] = F(2, 0); out_19[0] = F(2, 1); out_20[0] = F(2, 2); out_21[0] = F(2, 3); out_22[0] = F(2, 4); out_23[0] = F(2, 5); out_24[0] = F(2, 6); out_25[0] = F(2, 7); out_26[0] = F(2, 8); out_27[0] = F(3, 0); out_28[0] = F(3, 1); out_29[0] = F(3, 2); out_30[0] = F(3, 3); out_31[0] = F(3, 4); out_32[0] = F(3, 5); out_33[0] = F(3, 6); out_34[0] = F(3, 7); out_35[0] = F(3, 8); out_36[0] = F(4, 0); out_37[0] = F(4, 1); out_38[0] = F(4, 2); out_39[0] = F(4, 3); out_40[0] = F(4, 4); out_41[0] = F(4, 5); out_42[0] = F(4, 6); out_43[0] = F(4, 7); out_44[0] = F(4, 8); out_45[0] = F(5, 0); out_46[0] = F(5, 1); out_47[0] = F(5, 2); out_48[0] = F(5, 3); out_49[0] = F(5, 4); out_50[0] = F(5, 5); out_51[0] = F(5, 6); out_52[0] = F(5, 7); out_53[0] = F(5, 8); out_54[0] = F(6, 0); out_55[0] = F(6, 1); out_56[0] = F(6, 2); out_57[0] = F(6, 3); out_58[0] = F(6, 4); out_59[0] = F(6, 5); out_60[0] = F(6, 6); out_61[0] = F(6, 7); out_62[0] = F(6, 8); out_63[0] = F(7, 0); out_64[0] = F(7, 1); out_65[0] = F(7, 2); out_66[0] = F(7, 3); out_67[0] = F(7, 4); out_68[0] = F(7, 5); out_69[0] = F(7, 6); out_70[0] = F(7, 7); out_71[0] = F(7, 8); out_72[0] = F(8, 0); out_73[0] = F(8, 1); out_74[0] = F(8, 2); out_75[0] = F(8, 3); out_76[0] = F(8, 4); out_77[0] = F(8, 5); out_78[0] = F(8, 6); out_79[0] = F(8, 7); out_80[0] = F(8, 8);
    }


void point_triangle_ccd(float p0, float p1, float p2,
                            float t00, float t01, float t02,
                            float t10, float t11, float t12,
                            float t20, float t21, float t22,
                            float dp0, float dp1, float dp2,
                            float dt00, float dt01, float dt02,
                            float dt10, float dt11, float dt12,
                            float dt20, float dt21, float dt22,
                            float eta, float dist2, float* ret)
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
                    ret[0] = (float)toc;
                }
                else {
                    ret[0] = 1.0;
                }
            }
            ret[0] = (float)toc;
        }
        else {
            ret[0] = 1.0;
        }
    }

    void edge_edge_ccd(float a00, float a01, float a02,
                       float a10, float a11, float a12,
                       float b00, float b01, float b02,
                       float b10, float b11, float b12,
                       float da00, float da01, float da02,
                       float da10, float da11, float da12,
                       float db00, float db01, float db02,
                       float db10, float db11, float db12,
                       float eta, float dist2, float* ret)
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
                    ret[0] = (float)toc;
                }
                else {
                    ret[0] = 1.0;
                }
            }
            ret[0] = (float)toc;
        }
        else {
            ret[0] = 1.0;
        }
    }


};