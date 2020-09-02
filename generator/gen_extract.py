import sys

n = int(sys.argv[1])
for i in range(n):
    print('[', end='')
    for j in range(n):
        print('H[' + str(3 + i) + ', ' + str(3 + j) + ']', end=', ' if j != n - 1 else '], ')
print('')