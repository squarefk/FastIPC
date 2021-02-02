import numpy as np
from diff_test import *

def angle(v0, v1, v2):
    n1 = (v1 - v0)
    n2 = (v2 - v0)
    n1SqNorm = n1.dot(n1)
    n2SqNorm = n2.dot(n2)
    cos = n1.dot(n2) / np.sqrt(n1SqNorm * n2SqNorm)
    cos = np.max([-1., np.min([1., cos])])
    A = np.arccos(cos)
    return A

def compute_mHat(xp, xe0, xe1):
    e = xe1 - xe0
    mHat = xe0 + (xp - xe0).dot(e) / e.dot(e) * e - xp
    mHat /= np.linalg.norm(mHat)
    return mHat

def angel_gradient(v0, v1, v2):
    n1 = (v1 - v0)
    n2 = (v2 - v0)
    da_dv1 = compute_mHat(v2, v0, v1) / np.linalg.norm(n1)
    da_dv2 = compute_mHat(v1, v0, v2) / np.linalg.norm(n2)
    gradient = np.concatenate([-da_dv1 - da_dv2, da_dv1, da_dv2])
    return gradient

if __name__ == "__main__":
    def f(x):
        v0 = x[0:2]
        v1 = x[2:4]
        v2 = x[4:6]
        return angle(v0, v1, v2)
    def g(x):
        v0 = x[0:2]
        v1 = x[2:4]
        v2 = x[4:6]
        return angel_gradient(v0, v1, v2)

    x = np.random.random((6,))
    print(g(x))
    print(finite_gradient(x, f))