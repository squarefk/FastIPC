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
    dx = 2 * eps * (np.random.random(x.shape) - 0.5).astype(np.float64)
    x0 = x - dx
    x1 = x + dx
    f0 = f(x0)
    f1 = f(x1)
    try:
        g0, g1 = g(x0), g(x1)
    except:
        g0, g1 = np.zeros_like(x), np.zeros_like(x)
        g(x0, g0)
        g(x1, g1)
    true_value = np.abs(f1 - f0 - (g1 + g0).dot(dx)) / eps
    fake_value = np.abs(f1 - f0 - 2 * (g1 + g0).dot(dx)) / eps
    print("[Check Gradient] real_value: ", true_value, "[Check Gradient] fake_value: ", fake_value)
    return true_value / fake_value < pass_ratio

def check_jacobian(x, f, g, f_dim, eps = 1e-4, pass_ratio = 1e-3):
    dx = 2 * eps * (np.random.random(x.shape) - 0.5).astype(np.float64)
    x0 = x - dx
    x1 = x + dx
    f0 = np.zeros((f_dim, ))
    try:
        f0 = f(x0)
        f1 = f(x1)
    except:
        f0, f1 = np.zeros((f_dim, )), np.zeros((f_dim, ))
        f(x0, f0)
        f(x1, f1)

    try:
        g0 = g(x0)
        g1 = g(x1)
    except:
        g0, g1 = np.zeros((f_dim, x.size)), np.zeros((f_dim, x.size))
        g(x0, g0)
        g(x1, g1)
    true_value = np.linalg.norm(f1 - f0 - (g1 + g0).dot(dx)) / eps
    fake_value = np.linalg.norm(f1 - f0 - 2 * (g1 + g0).dot(dx)) / eps
    print("[Check Jacobian] real_value: ", true_value, "[Check Jacobian] fake_value: ", fake_value)
    return true_value / fake_value < pass_ratio