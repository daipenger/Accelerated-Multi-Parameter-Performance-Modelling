#include "LeastSquaresCPU.h"

#include "Eigen/Dense"
using namespace std;
using namespace Eigen;

/*
A in row-major
*/
void calculateLeastSquares(float* A, float* b, float* x, int rows, int cols) {
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>> eigenA(A, rows, cols);
    Map<Matrix<float, Dynamic, 1>> eigenb(b, rows, 1);
    Map<Matrix<float, Dynamic, 1>>(x, cols, 1) = eigenA.bdcSvd(ComputeThinU | ComputeThinV).solve(eigenb);
}
