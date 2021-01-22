import taichi as ti
import numpy as np

def finite_gradient(x, f, eps = 1e-6):
    grad = np.zeros_like(x)
    xx = x.copy()
    for d in range(x.size):
        xx[d] += eps
        grad[d] = f(xx)
        xx[d] -= 2 * eps
        grad[d] -= f(xx)
        xx[d] += eps
        grad[d] /= (2 * eps)
    return grad

def check_gradient(x, f, g, eps = 1e-6, pass_ratio = 1e-3):
    dx = 2 * eps * (np.random.random(x.shape) - 0.5);
    x0 = x - dx;
    x1 = x + dx;
    f0 = f(x0);
    f1 = f(x1);
    g0, g1 = np.zeros_like(x), np.zeros_like(x)
    g(x0, g0);
    g(x1, g1);
    true_value = np.abs(f1 - f0 - (g1 + g0).dot(dx)) / eps
    fake_value = np.abs(f1 - f0 - 2 * (g1 + g0).dot(dx)) / eps
    print("[Check Gradient] real_value: ", true_value, "[Check Gradient] fake_value: ", fake_value)
    return true_value / fake_value < pass_ratio