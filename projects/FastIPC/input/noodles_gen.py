import math

f = open('noodles.obj', 'w')


def build_one(sx, sy, dx, cnt, offset):
    for i in range(cnt):
        x, y = sx + dx * i, sy
        f.write('v %.4f %.4f 0\n' % (x, y))
        x, y = sx + dx * i, sy + dx * 0.5
        f.write('v %.4f %.4f 0\n' % (x, y))
    for i in range(cnt - 1):
        f.write('f %d %d %d\n' % (offset + i + i, offset + i + i + 2, offset + i + i + 1))
        f.write('f %d %d %d\n' % (offset + i + i + 1, offset + i + i + 2, offset + i + i + 3))


def build_plate(offset):
    x, y = 0.5, -0.01
    f.write('v %.4f %.4f 0\n' % (x * math.cos(math.pi / 6) - y * math.sin(math.pi / 6), x * math.sin(math.pi / 6) + y * math.cos(math.pi / 6)))
    x, y = 0.8, -0.01
    f.write('v %.4f %.4f 0\n' % (x * math.cos(math.pi / 6) - y * math.sin(math.pi / 6), x * math.sin(math.pi / 6) + y * math.cos(math.pi / 6)))
    x, y = 0.5, 0
    f.write('v %.4f %.4f 0\n' % (x * math.cos(math.pi / 6) - y * math.sin(math.pi / 6), x * math.sin(math.pi / 6) + y * math.cos(math.pi / 6)))
    x, y = 0.8, 0
    f.write('v %.4f %.4f 0\n' % (x * math.cos(math.pi / 6) - y * math.sin(math.pi / 6), x * math.sin(math.pi / 6) + y * math.cos(math.pi / 6)))
    f.write('f %d %d %d\n' % (offset, offset + 1, offset + 2))
    f.write('f %d %d %d\n' % (offset + 2, offset + 1, offset + 3))


def build_bot(offset):
    x, y = -2, -0.11
    f.write('v %.4f %.4f 0\n' % (x, y))
    x, y = 3, -0.11
    f.write('v %.4f %.4f 0\n' % (x, y))
    x, y = -2, -0.1
    f.write('v %.4f %.4f 0\n' % (x, y))
    x, y = 3, -0.1
    f.write('v %.4f %.4f 0\n' % (x, y))
    f.write('f %d %d %d\n' % (offset, offset + 1, offset + 2))
    f.write('f %d %d %d\n' % (offset + 2, offset + 1, offset + 3))



num = 100
for i in range(20):
    build_one(0, 0.42 + 0.017 * i, 0.01, num, num * 2 * i + 1)
build_plate(num * 2 * 20 + 1)
build_bot(num * 2 * 20 + 5)
print(num * 2 * 20 + 1, num * 2 * 20 + 2, num * 2 * 20 + 5, num * 2 * 20 + 6)

f.close()
