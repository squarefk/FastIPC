# void project_pd3(float F00, float F01, float F02,
#                  float F10, float F11, float F12,
#                  float F20, float F21, float F22,
#                  float* PF00, float* PF01, float* PF02,
#                  float* PF10, float* PF11, float* PF12,
#                  float* PF20, float* PF21, float* PF22)
# {
#     Eigen::Matrix<float, 3, 3> F;
#     F(0, 0) = F00; F(0, 1) = F01; F(0, 2) = F02;
#     F(1, 0) = F10; F(1, 1) = F11; F(1, 2) = F12;
#     F(2, 0) = F20; F(2, 1) = F21; F(2, 2) = F22;
#     JGSL::makePD(F);
#     PF00[0] = F(0, 0); PF01[0] = F(0, 1); PF02[0] = F(0, 2);
#     PF10[0] = F(1, 0); PF11[0] = F(1, 1); PF12[0] = F(1, 2);
#     PF20[0] = F(2, 0); PF21[0] = F(2, 1); PF22[0] = F(2, 2);
# }
import sys

n = int(sys.argv[1])
print('void project_pd_' + str(n) + '(', end='')
for i in range(n * n):
    print('float in_' + str(i) + ', ', end='')
for i in range(n * n):
    print('float* out_' + str(i), end='')
    if i != n * n - 1:
        print(', ', end='')
    else:
        print(')')
print('{')
print('Eigen::Matrix<float, ' + str(n) + ', ' + str(n) + '> F;')
for i in range(n):
    for j in range(n):
        print('F(' + str(i) + ', ' + str(j) + ') = in_' + str(i * n + j) + '; ', end='')
print('')
print('JGSL::makePD(F);')
for i in range(n):
    for j in range(n):
        print('out_' + str(i * n + j) + '[0] = F(' + str(i) + ', ' + str(j) + '); ', end='')
print('')
print('}')
