import matplotlib.pyplot as plt
import numpy as np

import triangle as tr
import random


def circle(n, r):
    pts = np.empty([0, 2], dtype=np.float)
    seg = np.empty([0, 2], dtype=np.int)
    now = 0
    while now < n:
        ran = random.randint(1, min(n - now, n // 3))
        now += ran
        alpha = 2 * np.pi / n * now
        radius = r * (0.3 + 0.7 * random.random())
        x = np.cos(alpha) * radius
        y = np.sin(alpha) * radius
        pts = np.vstack([pts, [x, y]])
    m = pts.shape[0]
    for i in range(m):
        seg = np.vstack([seg, [i, (i + 1) % m]])
    return pts, seg


all_pts = np.empty([0, 2], dtype=np.float)
all_seg = np.empty([0, 2], dtype=np.int)
for i in range(4):
    for j in range(5):
        pts, seg = circle(100, 1)
        all_seg = np.vstack([all_seg, seg + all_pts.shape[0]])
        all_pts = np.vstack([all_pts, pts * 1.75 + [i * 3 - 1.5, j * 3 + 4.5]])

A = dict(vertices=all_pts, segments=all_seg)
B = tr.triangulate(A, 'qpa0.04')
vertices = B['vertices']
triangles = B['triangles']

f = open('items.obj', 'w')
for [a, b] in vertices:
    f.write('v %.4f %.4f 0\n' % (a * 0.1, b * 0.1))
for [a, b, c] in triangles:
    f.write('f %d %d %d\n' % (a + 1, b + 1, c + 1))
f.close()
