#include "LeastSquaresGPU.h"
#include <cuda_runtime.h>
#include "gpuKernels.h"
#include "cublas_v2.h"
#include <stdio.h>

/*
pAd is the pointer structure (in device memory) to matrices A[i] for i <= batches:
A[i] are m x n matrices. in device memory. in column-major format.
b is a m x 1 matrix. in device memory.
m >= n (have to be overdetermined systems!)

return value: An array of length m*batches. The first n elements per batch
are the solution x[i] for the least squares problem A[i] * x[i] = b
*/
float* calculateLeastSquares(float** pAd, float* b, int m, int n, int batches) {
    cublasHandle_t handle;
    cublasStatus_t stat;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed. Status: %d\n", stat);
        return NULL;
    }

    float* bd;
    cudaMalloc((void**) &bd, sizeof(float) * m * batches);
    duplicateData(b, m, bd, m * batches);

    float** pB = (float**) malloc(sizeof(float*) * batches);
    float** pBd;
    cudaMalloc((void**) &pBd, sizeof(float*) * batches);

    for (int i = 0; i < batches; ++i) {
        pB[i] = &bd[m * i];
    }

    cudaMemcpy(pBd, pB, sizeof(float*) * batches, cudaMemcpyHostToDevice);
    free(pB);

    int info;
    stat = cublasSgelsBatched(handle, CUBLAS_OP_N, m, n, 1, pAd, m, pBd, m, &info, NULL, batches);

    if (info != 0) {
      printf("LeastSquaresGPU.cc: cublasSgelsBatched wrong parameter %d\n", -info);
    }

    cudaFree(pBd);
    cublasDestroy(handle);

    return bd;
}
