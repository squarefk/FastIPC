import math

f = open('spheres.obj', 'w')
x, y, r = 0, 0, 0.1
s = 20
for i in range(s):
    angle = math.pi * 2 / s * i - 0.5 * math.pi
    f.write('v %.5f %.5f %.5f\n' % (x + math.cos(angle) * r, y + math.sin(angle) * r, 0))
f.write('v %.5f %.5f %.5f\n' % (x, y, 0))
for i in range(s):
    f.write('f %d %d %d\n' % (s + 1, i + 1, (i + 1) % s + 1))
tmp = s + 1

x, y, r = 0, 0.201, 0.1
s = 20
for i in range(s):
    angle = math.pi * 2 / s * i - 0.5 * math.pi
    f.write('v %.5f %.5f %.5f\n' % (x + math.cos(angle) * r, y + math.sin(angle) * r, 0))
f.write('v %.5f %.5f %.5f\n' % (x, y, 0))
for i in range(s):
    f.write('f %d %d %d\n' % (tmp + s + 1, tmp + i + 1, tmp + (i + 1) % s + 1))

f.close()
