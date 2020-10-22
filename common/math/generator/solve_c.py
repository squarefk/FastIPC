# void solve_2(REAL in_0, REAL in_1, REAL in_2, REAL in_3, REAL rhs_0, REAL rhs_1, REAL* out_0, REAL* out_1)
# {
#     Eigen::Matrix<REAL, 2, 2> F;
#     F(0, 0) = in_0; F(0, 1) = in_1; F(1, 0) = in_2; F(1, 1) = in_3;
#     Eigen::Matrix<float, 2, 1> rhs;
#     rhs(0) = rhs_0; rhs(1) = rhs_1;
#     Eigen::Matrix<REAL, 2, 1> x = F.llt().solve(rhs);
#     out_0[0] = x(0); out_1[0] = x(1);
# }

import sys

n = int(sys.argv[1])
print('void solve_' + str(n) + '(', end='')
for i in range(n * n):
    print('REAL in_' + str(i) + ', ', end='')
for i in range(n):
    print('REAL rhs_' + str(i) + ', ', end='')
for i in range(n):
    print('REAL* out_' + str(i), end='')
    if i != n - 1:
        print(', ', end='')
    else:
        print(')')
print('{')
print('Eigen::Matrix<REAL, ' + str(n) + ', ' + str(n) + '> F;')
for i in range(n):
    for j in range(n):
        print('F(' + str(i) + ', ' + str(j) + ') = in_' + str(i * n + j) + '; ', end='')
print('')
print('Eigen::Matrix<REAL, ' + str(n) + ', 1> rhs;')
for i in range(n):
    print('rhs(' + str(i) + ') = rhs_' + str(i) + '; ', end='')
print('')
print('Eigen::Matrix<REAL, ' + str(n) + ', 1> x = F.llt().solve(rhs);')
for i in range(n):
    print('out_' + str(i) + '[0] = x(' + str(i) + '); ', end='')
print('')
print('}')
