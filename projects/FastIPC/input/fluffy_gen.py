import matplotlib.pyplot as plt
import numpy as np

import triangle as tr


def circle(n, m, r):
    pts = np.empty([0, 2], dtype=np.float)
    seg = np.empty([0, 2], dtype=np.int)
    for i in range(n):
        alpha = i * 2 * np.pi / n
        beta = (i // 2 * 2 + 0.5) * 2 * np.pi / n
        if i % 2 == 0:
            for j in range(m):
                x = np.cos(alpha) * r + np.cos(beta) * 0.12 * j
                y = np.sin(alpha) * r + np.sin(beta) * 0.12 * j
                pts = np.vstack([pts, [x, y]])
                seg = np.vstack([seg, [i * m + j, (i * m + j + 1) % (n * m)]])
        else:
            for j in range(m):
                x = np.cos(alpha) * r + np.cos(beta) * 0.12 * (m - 1 - j)
                y = np.sin(alpha) * r + np.sin(beta) * 0.12 * (m - 1 - j)
                pts = np.vstack([pts, [x, y]])
                seg = np.vstack([seg, [i * m + j, (i * m + j + 1) % (n * m)]])
    print(pts, '\n', seg)
    return pts, seg


pts, seg = circle(100, 20, 2)

A = dict(vertices=pts, segments=seg)
B = tr.triangulate(A, 'qpa0.05')
vertices = B['vertices']
triangles = B['triangles']

f = open('fluffy.obj', 'w')
for [a, b] in vertices:
    f.write('v %.4f %.4f 0\n' % (a * 0.1, b * 0.1))
for [a, b, c] in triangles:
    f.write('f %d %d %d\n' % (a + 1, b + 1, c + 1))
f.close()
