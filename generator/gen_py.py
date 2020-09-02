# @ti.func
# def project_pd3(F):
#     F00, F01, F02 = F(0, 0), F(0, 1), F(0, 2)
#     F10, F11, F12 = F(1, 0), F(1, 1), F(1, 2)
#     F20, F21, F22 = F(2, 0), F(2, 1), F(2, 2)
#     PF00, PF01, PF02 = 0.0, 0.0, 0.0
#     PF10, PF11, PF12 = 0.0, 0.0, 0.0
#     PF20, PF21, PF22 = 0.0, 0.0, 0.0
#     ti.external_func_call(func=so.project_pd3,
#                           args=(F00, F01, F02,
#                                 F10, F11, F12,
#                                 F20, F21, F22),
#                           outputs=(PF00, PF01, PF02,
#                                    PF10, PF11, PF12,
#                                    PF20, PF21, PF22))
#     return ti.Matrix([[PF00, PF01, PF02],
#                       [PF10, PF11, PF12],
#                       [PF20, PF21, PF22]])
import sys

n = int(sys.argv[1])
print('@ti.func')
print('def project_pd_' + str(n) + '(F):')
print('    ', end='')
for i in range(n * n):
    print('in_' + str(i), end=' = ' if i == n * n - 1 else ', ')
for i in range(n):
    for j in range(n):
        print('F[' + str(i) + ', ' + str(j) + ']', end='\n' if i == n - 1 and j == n - 1 else ', ')
print('    ', end='')
for i in range(n * n):
    print('out_' + str(i), end=' = ' if i == n * n - 1 else ', ')
for i in range(n):
    for j in range(n):
        print('0.0', end='\n' if i == n - 1 and j == n - 1 else ', ')
print('    ti.external_func_call(func=so.project_pd_' + str(n) + ', args=(', end='')
for i in range(n * n):
    print('in_' + str(i), end='), ' if i == n * n - 1 else ', ')
print('outputs=(', end='')
for i in range(n * n):
    print('out_' + str(i), end='))\n' if i == n * n - 1 else ', ')
print('    return ti.Matrix([', end='')
for i in range(n):
    print('[', end='')
    for j in range(n):
        print('out_' + str(i * n + j), end='' if j == n - 1 else ', ')
    print(']', end='])' if i == n - 1 else ', ')
print('')
