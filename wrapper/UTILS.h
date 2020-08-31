#pragma once

#include <Eigen/Eigen>

namespace JGSL {

template<class T, int size>
void makePD(Eigen::Matrix<T, size, size>& symMtr)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
    if (eigenSolver.eigenvalues()[0] >= 0) {
        return;
    }
    Eigen::DiagonalMatrix<T, size> D(eigenSolver.eigenvalues());
    int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
    for (int i = 0; i < rows; ++i) {
        if (D.diagonal()[i] < 0) {
            D.diagonal()[i] = 0;
        }
        else {
            break;
        }
    }
    symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}

}

