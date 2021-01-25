import taichi as ti
from simplex_volume import *
from diff_test import *

x = ti.Vector.field(3, ti.float64, shape=3)

x.from_numpy(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]))
thickness = 0.02345
codim = 2
dim = 3
mu = 0.452
la = 0.251

@ti.kernel
def f(v: ti.ext_arr(), arr: ti.ext_arr()):
    X1 = ti.Vector([v[0], v[1], v[2]])
    X2 = ti.Vector([v[3], v[4], v[5]])
    X3 = ti.Vector([v[6], v[7], v[8]])
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    AB = X2 - X1
    AC = X3 - X1
    ab = x2 - x1
    ac = x3 - x1
    Tx = ti.Matrix.cols([ab, ac])
    TX = ti.Matrix.cols([AB, AC])
    A = Tx.transpose() @ Tx
    IA = A.inverse()
    B = TX.transpose() @ TX
    IB = B.inverse()
    vol = thickness * (AB.cross(AC)).norm() / 2
    
    lnJ = 0.5 * ti.log(A.determinant() * IB.determinant())
    de_div_dA = ti.Vector([0.0, 0.0, 0.0, 0.0])
    for i in ti.static(range(2)):
        for j in ti.static(range(2)):
            de_div_dA[j * codim + i] = ((0.5 * mu * IB[i,j] + 0.5 * (-mu + la * lnJ) * IA[i,j]))
    
    Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dA_div_dx = ti.Matrix.rows([Z, Z, Z, Z])
    for i in ti.static(range(3)):
        dA_div_dx[0, 3 + i] += 2.0 * (x2[i] - x1[i])
        dA_div_dx[0, 0 + i] -= 2.0 * (x2[i] - x1[i])
        dA_div_dx[1, 6 + i] += (x2[i] - x1[i])
        dA_div_dx[1, 3 + i] += (x3[i] - x1[i])
        dA_div_dx[1, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
        dA_div_dx[2, 6 + i] += (x2[i] - x1[i])
        dA_div_dx[2, 3 + i] += (x3[i] - x1[i])
        dA_div_dx[2, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
        dA_div_dx[3, 6 + i] += 2.0 * (x3[i] - x1[i])
        dA_div_dx[3, 0 + i] -= 2.0 * (x3[i] - x1[i])
    
    grad = vol * dA_div_dx.transpose() @ de_div_dA
    for i in ti.static(range(9)):
        arr[i] = grad[i]

@ti.kernel
def df(v: ti.ext_arr(), arr: ti.ext_arr()):
    X1 = ti.Vector([v[0], v[1], v[2]])
    X2 = ti.Vector([v[3], v[4], v[5]])
    X3 = ti.Vector([v[6], v[7], v[8]])
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    AB = X2 - X1
    AC = X3 - X1
    ab = x2 - x1
    ac = x3 - x1
    Tx = ti.Matrix.cols([ab, ac])
    TX = ti.Matrix.cols([AB, AC])
    A = Tx.transpose() @ Tx
    IA = A.inverse()
    B = TX.transpose() @ TX
    IB = B.inverse()
    vol = thickness * (AB.cross(AC)).norm() / 2

    lnJ = 0.5 * ti.log(A.determinant() * IB.determinant())
    dv_div_dX = thickness * simplex_volume_gradient(X1, X2, X3)
    de_div_dA = ti.Vector([0.,0.,0.,0.])
    for i in ti.static(range(2)):
        for j in ti.static(range(2)):
            de_div_dA[j * codim + i] = ((0.5 * mu * IB[i,j] + 0.5 * (-mu + la * lnJ) * IA[i,j]))
    Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dA_div_dx = ti.Matrix.rows([Z, Z, Z, Z])
    dB_div_dX = ti.Matrix.rows([Z, Z, Z, Z])
    for i in ti.static(range(3)):
        dA_div_dx[0, 3 + i] += 2.0 * (x2[i] - x1[i])
        dA_div_dx[0, 0 + i] -= 2.0 * (x2[i] - x1[i])
        dA_div_dx[1, 6 + i] += (x2[i] - x1[i])
        dA_div_dx[1, 3 + i] += (x3[i] - x1[i])
        dA_div_dx[1, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
        dA_div_dx[2, 6 + i] += (x2[i] - x1[i])
        dA_div_dx[2, 3 + i] += (x3[i] - x1[i])
        dA_div_dx[2, 0 + i] += - (x2[i] - x1[i]) - (x3[i] - x1[i])
        dA_div_dx[3, 6 + i] += 2.0 * (x3[i] - x1[i])
        dA_div_dx[3, 0 + i] -= 2.0 * (x3[i] - x1[i])

        dB_div_dX[0, 3 + i] += 2.0 * (X2[i] - X1[i])
        dB_div_dX[0, 0 + i] -= 2.0 * (X2[i] - X1[i])
        dB_div_dX[1, 6 + i] += (X2[i] - X1[i])
        dB_div_dX[1, 3 + i] += (X3[i] - X1[i])
        dB_div_dX[1, 0 + i] += - (X2[i] - X1[i]) - (X3[i] - X1[i])
        dB_div_dX[2, 6 + i] += (X2[i] - X1[i])
        dB_div_dX[2, 3 + i] += (X3[i] - X1[i])
        dB_div_dX[2, 0 + i] += - (X2[i] - X1[i]) - (X3[i] - X1[i])
        dB_div_dX[3, 6 + i] += 2.0 * (X3[i] - X1[i])
        dB_div_dX[3, 0 + i] -= 2.0 * (X3[i] - X1[i])
        
    de_div_dx = dA_div_dx.transpose() @ de_div_dA

    # first term
    hessian = dv_div_dX @ de_div_dx.transpose()

    # second term
    Z4 = ti.Vector([0.,0.,0.,0.])
    dbinv_div_db = ti.Matrix.rows([Z4, Z4, Z4, Z4])
    for m in ti.static(range(2)):
        for n in ti.static(range(2)):
            for i in ti.static(range(2)):
                for j in ti.static(range(2)): 
                    dbinv_div_db[n * codim + m, j * codim + i] = - IB[m, i] * IB[j, n]
    
    d2e_divA_divB = 0.5 * mu * dbinv_div_db
    for m in ti.static(range(2)):
        for n in ti.static(range(2)):
            for i in ti.static(range(2)):
                for j in ti.static(range(2)):
                    d2e_divA_divB[n * codim + m, j * codim + i] -= 0.25 * la * IA[m, n] * IB[j, i]
    hessian += vol * dB_div_dX.transpose() @ d2e_divA_divB.transpose() @ dA_div_dx

    for i in ti.static(range(9)):
        for j in ti.static(range(9)):
            arr[i, j] = hessian[j, i]


if __name__ == '__main__':
    x.from_numpy(np.random.random((3, 3)).astype(np.float64))
    v = x.to_numpy().flatten() * 10.
    check_jacobian(v, f, df, v.size, eps=1e-2)
    check_jacobian(v, f, df, v.size, eps=1e-2)
    check_jacobian(v, f, df, v.size, eps=1e-2)