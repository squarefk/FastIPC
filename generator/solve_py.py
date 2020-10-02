# @ti.func
# def linear_solve_2(F, rhs):
#     in_0, in_1, in_2, in_3, in_4, in_5 = F[0, 0], F[0, 1], F[1, 0], F[1, 1], rhs[0], rhs[1]
#     out_0, out_1 = 0.0, 0.0
#     ti.external_func_call(func=so.solve_2, args=(in_0, in_1, in_2, in_3, in_4, in_5), outputs=(out_0, out_1))
#     return ti.Vector([out_0, out_1])

import sys

n = int(sys.argv[1])
print('@ti.func')
print('def solve_' + str(n) + '(F, rhs):')
print('    ', end='')
for i in range(n * n):
    print('in_' + str(i), end=', ')
for i in range(n):
    print('in_' + str(n * n + i), end=' = ' if i == n - 1 else ', ')
for i in range(n):
    for j in range(n):
        print('F[' + str(i) + ', ' + str(j) + ']', end=', ')
for i in range(n):
    print('rhs[' + str(i) + ']', end='\n' if i == n - 1 else ', ')
print('    ', end='')
for i in range(n):
    print('out_' + str(i), end=' = ' if i == n - 1 else ', ')
for i in range(n):
    print('0.0', end='\n' if i == n - 1 else ', ')
print('    ti.external_func_call(func=so.solve_' + str(n) + ', args=(', end='')
for i in range(n * n + n):
    print('in_' + str(i), end='), ' if i == n * n + n- 1 else ', ')
print('outputs=(', end='')
for i in range(n):
    print('out_' + str(i), end='))\n' if i == n - 1 else ', ')
print('    return ti.Vector([', end='')
for i in range(n):
    print('out_' + str(i), end=']) ' if i == n - 1 else ', ')
print('')
