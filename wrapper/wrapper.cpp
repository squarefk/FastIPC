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

    void project_pd(float F00, float F01, float F02, float F03, float F04, float F05,
                    float F10, float F11, float F12, float F13, float F14, float F15,
                    float F20, float F21, float F22, float F23, float F24, float F25,
                    float F30, float F31, float F32, float F33, float F34, float F35,
                    float F40, float F41, float F42, float F43, float F44, float F45,
                    float F50, float F51, float F52, float F53, float F54, float F55, float diagonal,
                    float* PF00, float* PF01, float* PF02, float* PF03, float* PF04, float* PF05,
                    float* PF10, float* PF11, float* PF12, float* PF13, float* PF14, float* PF15,
                    float* PF20, float* PF21, float* PF22, float* PF23, float* PF24, float* PF25,
                    float* PF30, float* PF31, float* PF32, float* PF33, float* PF34, float* PF35,
                    float* PF40, float* PF41, float* PF42, float* PF43, float* PF44, float* PF45,
                    float* PF50, float* PF51, float* PF52, float* PF53, float* PF54, float* PF55)
    {
        Eigen::Matrix<float, 6, 6> F;
        F(0, 0) = F00; F(0, 1) = F01; F(0, 2) = F02; F(0, 3) = F03; F(0, 4) = F04; F(0, 5) = F05;
        F(1, 0) = F10; F(1, 1) = F11; F(1, 2) = F12; F(1, 3) = F13; F(1, 4) = F14; F(1, 5) = F15;
        F(2, 0) = F20; F(2, 1) = F21; F(2, 2) = F22; F(2, 3) = F23; F(2, 4) = F24; F(2, 5) = F25;
        F(3, 0) = F30; F(3, 1) = F31; F(3, 2) = F32; F(3, 3) = F33; F(3, 4) = F34; F(3, 5) = F35;
        F(4, 0) = F40; F(4, 1) = F41; F(4, 2) = F42; F(4, 3) = F43; F(4, 4) = F44; F(4, 5) = F45;
        F(5, 0) = F50; F(5, 1) = F51; F(5, 2) = F52; F(5, 3) = F53; F(5, 4) = F54; F(5, 5) = F55;
        JGSL::makePD(F);
        for (int i = 0; i < 6; ++i)
            F(i, i) += diagonal;
        F = F.inverse();
        PF00[0] = F(0, 0); PF01[0] = F(0, 1); PF02[0] = F(0, 2); PF03[0] = F(0, 3); PF04[0] = F(0, 4); PF05[0] = F(0, 5);
        PF10[0] = F(1, 0); PF11[0] = F(1, 1); PF12[0] = F(1, 2); PF13[0] = F(1, 3); PF14[0] = F(1, 4); PF15[0] = F(1, 5);
        PF20[0] = F(2, 0); PF21[0] = F(2, 1); PF22[0] = F(2, 2); PF23[0] = F(2, 3); PF24[0] = F(2, 4); PF25[0] = F(2, 5);
        PF30[0] = F(3, 0); PF31[0] = F(3, 1); PF32[0] = F(3, 2); PF33[0] = F(3, 3); PF34[0] = F(3, 4); PF35[0] = F(3, 5);
        PF40[0] = F(4, 0); PF41[0] = F(4, 1); PF42[0] = F(4, 2); PF43[0] = F(4, 3); PF44[0] = F(4, 4); PF45[0] = F(4, 5);
        PF50[0] = F(5, 0); PF51[0] = F(5, 1); PF52[0] = F(5, 2); PF53[0] = F(5, 3); PF54[0] = F(5, 4); PF55[0] = F(5, 5);
    }

    void project_pd64(float F00, float F01, float F02, float F03, float F04, float F05,
                    float F10, float F11, float F12, float F13, float F14, float F15,
                    float F20, float F21, float F22, float F23, float F24, float F25,
                    float F30, float F31, float F32, float F33, float F34, float F35,
                    float F40, float F41, float F42, float F43, float F44, float F45,
                    float F50, float F51, float F52, float F53, float F54, float F55, float diagonal,
                    float* PF00, float* PF01, float* PF02, float* PF03,
                    float* PF10, float* PF11, float* PF12, float* PF13,
                    float* PF20, float* PF21, float* PF22, float* PF23,
                    float* PF30, float* PF31, float* PF32, float* PF33)
    {
        Eigen::Matrix<float, 4, 4> F;
        F(0, 0) = F22; F(0, 1) = F23; F(0, 2) = F24; F(0, 3) = F25;
        F(1, 0) = F32; F(1, 1) = F33; F(1, 2) = F34; F(1, 3) = F35;
        F(2, 0) = F42; F(2, 1) = F43; F(2, 2) = F44; F(2, 3) = F45;
        F(3, 0) = F52; F(3, 1) = F53; F(3, 2) = F54; F(3, 3) = F55;
        JGSL::makePD(F);
        for (int i = 0; i < 4; ++i)
            F(i, i) += diagonal;
        F = F.inverse().eval();
        PF00[0] = F(0, 0); PF01[0] = F(0, 1); PF02[0] = F(0, 2); PF03[0] = F(0, 3);
        PF10[0] = F(1, 0); PF11[0] = F(1, 1); PF12[0] = F(1, 2); PF13[0] = F(1, 3);
        PF20[0] = F(2, 0); PF21[0] = F(2, 1); PF22[0] = F(2, 2); PF23[0] = F(2, 3);
        PF30[0] = F(3, 0); PF31[0] = F(3, 1); PF32[0] = F(3, 2); PF33[0] = F(3, 3);
    }

    void project_pd3(float F00, float F01, float F02,
                     float F10, float F11, float F12,
                     float F20, float F21, float F22,
                     float* PF00, float* PF01, float* PF02,
                     float* PF10, float* PF11, float* PF12,
                     float* PF20, float* PF21, float* PF22)
    {
        Eigen::Matrix<float, 3, 3> F;
        F(0, 0) = F00; F(0, 1) = F01; F(0, 2) = F02;
        F(1, 0) = F10; F(1, 1) = F11; F(1, 2) = F12;
        F(2, 0) = F20; F(2, 1) = F21; F(2, 2) = F22;
        JGSL::makePD(F);
        PF00[0] = F(0, 0); PF01[0] = F(0, 1); PF02[0] = F(0, 2);
        PF10[0] = F(1, 0); PF11[0] = F(1, 1); PF12[0] = F(1, 2);
        PF20[0] = F(2, 0); PF21[0] = F(2, 1); PF22[0] = F(2, 2);
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