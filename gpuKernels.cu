#include "PARAMETERS.h"
#include "gpuKernels.h"
#include "common.h"
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix.h"
#include "LeastSquaresGPU.h"
#include "cublas_v2.h"

#define MAX_DIMENSIONS 5

#define SELECT_A 0
#define SELECT_B 1
#define SELECT_C 2
#define SELECT_D 3
#define SELECT_E 4

#define FUNCTION_MASK_A 0b1
#define FUNCTION_MASK_B 0b10
#define FUNCTION_MASK_C 0b100
#define FUNCTION_MASK_D 0b1000
#define FUNCTION_MASK_E 0b10000

const float I[sizeI] = {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3};
const float J[sizeJ] = {0, 1, 2};
__constant__ float I_gpu[sizeI];
__constant__ float J_gpu[sizeJ];

// TODO: Replace 0 * nan = nan behaviour in the kernels with 0 * nan = 0. (or make log2f return 0 for 0)
// Cause: Measurement coordinates can be 0
#include "lsMatricesKernelsColumnMajor"
#include "rss_smape_kernels"
__global__ void dupKernel(const float* __restrict__ in, const int inLen, float* __restrict__ out, const int outLen) {
    uint64_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (tid >= outLen)
		return;
	out[tid] = in[tid % inLen];
}

void duplicateData(const float* __restrict__ in, const int inLen, float* __restrict__ out, const int outLen) {
	dupKernel<<<(outLen + 31) >> 5, 32>>>(in, inLen, out, outLen);
}

__global__ void swapKernel(float *elements1, float *elements2, size_t len) {
    uint64_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid >= len)
    	return;
	__shared__ float s[32];
	s[threadIdx.x] = elements1[tid];
	elements1[tid] = elements2[tid];
	elements2[tid] = s[threadIdx.x];
}

void swapRows(GPUMatrix &Md, int x, int y) {
	swapKernel<<<(Md.width + 31) >> 5, 32>>>((float*)((char*)(Md.elements) + (x * Md.pitch)),
		(float*)((char*)(Md.elements) + (y * Md.pitch)), Md.width);
}

// Preallocated memory for calculateBestFunction -> Not usable for cuda streams and not thread safe.
// Why don't we use streams? -> We do not have any bigger amount of transfers, we
// by design are bound by arithmetic. streams would not help much.
// But: Safes a lot of runtime and stalls on host to just reuse this memory
	float* lsMatricesD;
	char* coefD;
	float* rssSmape;
	float** pAd;
	float* bd;

// Memory layout of the return value:
// Data type: float^nCoefficients | float | uint64_t
// Semantics: coefficients | RSS/SMAPE | function id
char* calculateBestFunction
(int nCoefficients, int nMeasurements, uint64_t nHypothesis, int nDimensions,
	float* b, uint64_t functionMask, GPUMatrix &Md,
	void (*functionPtr)(selects, uint64_t, int, float*, size_t, float* __restrict__), selects s,
	bool doBatched = false, uint64_t batch = 0,
	void (*batchedFunctionPtr)(uint64_t, uint64_t, int, float*, size_t, float* __restrict__) = NULL) {

	size_t coefPitch = nCoefficients * sizeof(float);

	// Calculate RSS/SMAPE based on cross-validation, if cross-validation is on
	if (USE_CROSSVALIDATION) {
	// reset rssSmape to 0 immediately
	cudaMemset(rssSmape, 0, sizeof(float) * nHypothesis);
	
	for (int leaveOneOut = 0; leaveOneOut < Md.height; ++leaveOneOut) {
		// Swap the leaveOnOut'th measurement with the last measurement.
		// The last row is then the leaved one out "test set".
		swapRows(Md, leaveOneOut, Md.height - 1);
		// also have to swap the solution vector at these locations:
		{
			float bSwap = b[leaveOneOut];
			b[leaveOneOut] = b[Md.height - 1];
			b[Md.height - 1] = bSwap;
		}

		// CALCULATE MATRICES FOR LEAST SQUARES SOLVING
		if (doBatched)
			(*batchedFunctionPtr)<<<(nHypothesis + 511) >> 9, 512>>>(batch, nHypothesis, nMeasurements - 1, lsMatricesD, Md.pitch, Md.elements);
		else
			(*functionPtr)<<<(nHypothesis + 511) >> 9, 512>>>(s, nHypothesis, nMeasurements - 1, lsMatricesD, Md.pitch, Md.elements);
		// CALCULATE LEAST SQUARES
		// Memory layout of coefD:
		// (float^nCoefficients)^nHypothesis
		{
			// Make that cuBLAS pointer structure to the matrices for least squares solving
			float** pA = (float**) malloc(sizeof(float*) * nHypothesis);
			for (uint64_t i = 0; i < nHypothesis; ++i) {
				pA[i] = &lsMatricesD[nCoefficients * (nMeasurements - 1) * i];
			}
			// and that must be in device memory
			cudaMemcpy(pAd, pA, sizeof(float*) * nHypothesis, cudaMemcpyHostToDevice);
			free(pA);

			// solution vector b must be in device memory also
			cudaMemcpy(bd, b, sizeof(float) * (nMeasurements - 1), cudaMemcpyHostToDevice);

			float* result = calculateLeastSquares(pAd, bd, nMeasurements - 1, nCoefficients, nHypothesis);

			// Copy to least squares solutions to coefD (why don't we just keep result?: result has
			// uncomfortable to work with pitch.  see the implementation of calculateLeastSquares and cuBLAS)
			cudaMemcpy2D(coefD, coefPitch, result, (nMeasurements - 1) * sizeof(float), nCoefficients * sizeof(float), nHypothesis, cudaMemcpyDeviceToDevice);
			cudaFree(result);
		}
		// CALCULATE RSS/SMAPE
		if (USE_SMAPE) {
			kernelSMAPE<<<(nHypothesis + 255) >> 8, 256>>>(nHypothesis, nCoefficients, nDimensions, nMeasurements - 1, functionMask, coefD, coefPitch, rssSmape, (float*)((char*)Md.elements + (Md.height - 1) * Md.pitch), batch);
		} else {
			kernelRSS<<<(nHypothesis + 255) >> 8, 256>>>(nHypothesis, nCoefficients, nDimensions, nMeasurements - 1, functionMask, coefD, coefPitch, rssSmape, (float*)((char*)Md.elements + (Md.height - 1) * Md.pitch), batch);
		}

		// swap back that stuff
		swapRows(Md, leaveOneOut, Md.height - 1);
		{
			float bSwap = b[leaveOneOut];
			b[leaveOneOut] = b[Md.height - 1];
			b[Md.height - 1] = bSwap;
		}
	}
	}

	// Calculate the actual model for all measurements

	// CALCULATE MATRICES FOR LEAST SQUARES SOLVING
	// Needs at least Kepler for up to 2^31 - 1 blocks in the x dimension of the grid
	// Let's just design it for that, otherwise it's still doable, but becomes cumbersome
	if (doBatched)
		(*batchedFunctionPtr)<<<(nHypothesis + 511) >> 9, 512>>>(batch, nHypothesis, nMeasurements, lsMatricesD, Md.pitch, Md.elements);
	else
		(*functionPtr)<<<(nHypothesis + 511) >> 9, 512>>>(s, nHypothesis, nMeasurements, lsMatricesD, Md.pitch, Md.elements);

	// CALCULATE LEAST SQUARES
	// Memory layout of coefD:
	// (float^nCoefficients)^nHypothesis
	{
		// Make that cuBLAS pointer structure to the matrices for least squares solving
		float** pA = (float**) malloc(sizeof(float*) * nHypothesis);
		for (uint64_t i = 0; i < nHypothesis; ++i) {
			pA[i] = &lsMatricesD[nCoefficients * nMeasurements * i];
		}
		// and that must be in device memory
		cudaMemcpy(pAd, pA, sizeof(float*) * nHypothesis, cudaMemcpyHostToDevice);
		free(pA);

		// solution vector b must be in device memory also
		cudaMemcpy(bd, b, sizeof(float) * nMeasurements, cudaMemcpyHostToDevice);

		float* result = calculateLeastSquares(pAd, bd, nMeasurements, nCoefficients, nHypothesis);

		// Copy to least squares solutions to coefD (why don't we just keep result?: result has
		// uncomfortable to work with pitch.  see the implementation of calculateLeastSquares and cuBLAS)
		cudaMemcpy2D(coefD, coefPitch, result, nMeasurements * sizeof(float), nCoefficients * sizeof(float), nHypothesis, cudaMemcpyDeviceToDevice);
		cudaFree(result);
	}

	// If we didn't use cross-validation to calculate RSS/SMAPE, do it now with the actual model
	if (!USE_CROSSVALIDATION) {
	// CALCULATE RSS/SMAPE
	// reset rssSmape to 0
	cudaMemset(rssSmape, 0, sizeof(float) * nHypothesis);
	for (int i = 0; i < Md.height; ++i) {
		if (USE_SMAPE) {
			kernelSMAPE<<<(nHypothesis + 255) >> 8, 256>>>(nHypothesis, nCoefficients, nDimensions, nMeasurements, functionMask, coefD, coefPitch, rssSmape, (float*)((char*)Md.elements + i * Md.pitch), batch);
		} else {
			kernelRSS<<<(nHypothesis + 255) >> 8, 256>>>(nHypothesis, nCoefficients, nDimensions, nMeasurements, functionMask, coefD, coefPitch, rssSmape, (float*)((char*)Md.elements + i * Md.pitch), batch);
		}
	}
	}

	// FIND THE BEST FUNCTION
	int bestFunction;
	{
		cublasHandle_t handle;
    	cublasCreate(&handle);
    	cublasIsamin(handle, nHypothesis, rssSmape, 1, &bestFunction);
    	cublasDestroy(handle);
    	// because cublasIsamin uses 1-indexing
    	--bestFunction;
	}

	// Memory layout: float^nCoefficients | float | uint64_t
	char* result = (char*) malloc(coefPitch + sizeof(float) + sizeof(uint64_t));
	// copy coefficients
	cudaMemcpy(result, coefD + (coefPitch * bestFunction), coefPitch, cudaMemcpyDeviceToHost);
	// copy rss/smape
	cudaMemcpy(&result[coefPitch], &rssSmape[bestFunction], sizeof(float), cudaMemcpyDeviceToHost);
	// set id
	*((uint64_t*) (&result[coefPitch + sizeof(float)])) = bestFunction + batch;

	return result;
}

Function* getBestFunctionAndDestroyOther(Function* x, Function* y) {
	#ifdef DRDEBUG
		y->print();
	#endif
	if (x->getError() > y->getError()) {
		delete x;
		return y;
	}
	delete y;
	return x;
}

Function* process(const CPUMatrix &M) {
	// Number of total threads:
	// For 2 dimensions: 4 * (sizeI * sizeJ)^2 = 6,084
	// For 3 dimensions: 23 * (sizeI * sizeJ)^3 = 1,364,337
	// For x dimensions (x>=4): 2 * (sizeI * sizeJ)^x
	// WARNING: Currently we can go up to 5 dimensions (including). For 6 and beyond
	// we exceed the grid x dimensions (2^31 - 1).

	cudaMemcpyToSymbol(I_gpu, I, sizeof(float) * sizeI, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(J_gpu, J, sizeof(float) * sizeJ, 0, cudaMemcpyHostToDevice);

	GPUMatrix Md = matrix_alloc_gpu(M.width, M.height);
	matrix_upload(M, Md);

	// See header for explanation
	const int nDimensions = M.width - 1;
	const int nMeasurements = M.height;
	// These are the number of i,j combinations for each type of hyptothesis
	const uint64_t nThreads = ipow(sizeI * sizeJ, nDimensions);

	// b contains the measured runtimes/values. It's the vector in A*x = b for least squares solving.
	float* b = (float*) malloc(sizeof(float) * nMeasurements);
	for (int i = 0; i < nMeasurements; ++i) {
		b[i] = M.elements[i * M.width + (M.width - 1)];
	}

	// Currently maxNCoefficients is nDimensions + 1, but has to adjust, if this changes in the future someday
	int maxNCoefficients = nDimensions + 1;

	// 4 and higher dimensions do batched
	// processing in order to not exceed VRAM:
	// with the current sizeI and sizeJ, we choose 177957 threads per batch
	// WARNING: 177957 must divide nThreads in the case for 4 and higher dimensions!
	uint64_t batchSize = min(nThreads, ipow(sizeI * sizeJ, 3) * sizeJ);

	size_t lsMatricesSize = maxNCoefficients * nMeasurements * sizeof(float) * batchSize;
	cudaMalloc((void**) &lsMatricesD, lsMatricesSize);
	size_t coefSize = maxNCoefficients * sizeof(float) * batchSize;
	cudaMalloc((void**) &coefD, coefSize);
	cudaMalloc((void**) &rssSmape, sizeof(float) * batchSize);
    cudaMalloc((void**) &pAd, sizeof(float*) * batchSize);
	cudaMalloc((void**) &bd, sizeof(float) * nMeasurements);

	//#ifdef DRDEBUG
		printf("#Dimensions: %d, #Measurements: %d, #ThreadsPerHypothesisType: %ld\n", nDimensions, nMeasurements, nThreads);
	//#endif

	Function* bestFunction = NULL;

	if (nDimensions == 2) {
		int nCoefficients = 2;
		char* bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B, Md, &kernelAxB, {SELECT_A,SELECT_B,0,0,0,0});
		bestFunction = new Function(nCoefficients, bf, "a*b", 0);

		nCoefficients = 3;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | (FUNCTION_MASK_B << 2), Md, &kernelApB, {SELECT_A,SELECT_B,0,0,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a+b", 1));

		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, (FUNCTION_MASK_A | FUNCTION_MASK_B) | (FUNCTION_MASK_A << 2), Md, &kernelAxBpC, {SELECT_A,SELECT_B,SELECT_A,0,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b + a", 2));

		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, (FUNCTION_MASK_A | FUNCTION_MASK_B) | (FUNCTION_MASK_B << 2), Md, &kernelAxBpC, {SELECT_A,SELECT_B,SELECT_B,0,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b + b", 3));
	} else if (nDimensions == 3) {
		int nCoefficients = 2;
		char* bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C, Md, &kernelAxBxC, {SELECT_A,SELECT_B,SELECT_C,0,0,0});
		bestFunction = new Function(nCoefficients, bf, "a*b*c", 0);

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | (FUNCTION_MASK_B << 3) | (FUNCTION_MASK_C << 6), Md, &kernelApBpC, {SELECT_A,SELECT_B,SELECT_C,0,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a + b + c", 1));

		nCoefficients = 3;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | (FUNCTION_MASK_A << 3), Md, &kernelAxBxCpD, {SELECT_A,SELECT_B,SELECT_C,SELECT_A,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c + a", 2));

		nCoefficients = 3;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | (FUNCTION_MASK_B << 3), Md, &kernelAxBxCpD, {SELECT_A,SELECT_B,SELECT_C,SELECT_B,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c + b", 3));

		nCoefficients = 3;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | (FUNCTION_MASK_C << 3), Md, &kernelAxBxCpD, {SELECT_A,SELECT_B,SELECT_C,SELECT_C,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c + c", 4));

		nCoefficients = 3;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | ((FUNCTION_MASK_A | FUNCTION_MASK_B) << 3), Md, &kernelAxBxCpDxE, {SELECT_A,SELECT_B,SELECT_C,SELECT_A,SELECT_B,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c + a*b", 5));

		nCoefficients = 3;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | ((FUNCTION_MASK_B | FUNCTION_MASK_C) << 3), Md, &kernelAxBxCpDxE, {SELECT_A,SELECT_B,SELECT_C,SELECT_B,SELECT_C,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c + b*c", 6));

		nCoefficients = 3;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | ((FUNCTION_MASK_A | FUNCTION_MASK_C) << 3), Md, &kernelAxBxCpDxE, {SELECT_A,SELECT_B,SELECT_C,SELECT_A,SELECT_C,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c + a*c", 7));

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | ((FUNCTION_MASK_A | FUNCTION_MASK_B) << 3) | (FUNCTION_MASK_A << 6), Md, &kernelAxBxCpDxEpF, {SELECT_A,SELECT_B,SELECT_C,SELECT_A,SELECT_B,SELECT_A});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c + a*b + a", 8));

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | ((FUNCTION_MASK_B | FUNCTION_MASK_C) << 3) | (FUNCTION_MASK_B << 6), Md, &kernelAxBxCpDxEpF, {SELECT_A,SELECT_B,SELECT_C,SELECT_B,SELECT_C,SELECT_B});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c + b*c + b", 9));

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | ((FUNCTION_MASK_A | FUNCTION_MASK_C) << 3) | (FUNCTION_MASK_C << 6), Md, &kernelAxBxCpDxEpF, {SELECT_A,SELECT_B,SELECT_C,SELECT_A,SELECT_C,SELECT_C});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c + a*c + c", 10));

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | ((FUNCTION_MASK_A) << 3) | (FUNCTION_MASK_B << 6), Md, &kernelAxBxCpDxEpF, {SELECT_A,SELECT_B,SELECT_C,SELECT_A,SELECT_B,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c + a + b", 11));

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | ((FUNCTION_MASK_B) << 3) | (FUNCTION_MASK_C << 6), Md, &kernelAxBxCpDxEpF, {SELECT_A,SELECT_B,SELECT_C,SELECT_B,SELECT_C,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c + b + c", 12));

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | ((FUNCTION_MASK_A) << 3) | (FUNCTION_MASK_C << 6), Md, &kernelAxBxCpDxEpF, {SELECT_A,SELECT_B,SELECT_C,SELECT_A,SELECT_C,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c + a + c", 13));

		nCoefficients = 3;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | (FUNCTION_MASK_C << 3), Md, &kernelAxBpC, {SELECT_A,SELECT_B,SELECT_C,0,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b + c", 14));

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | ((FUNCTION_MASK_C) << 3) | (FUNCTION_MASK_B << 6), Md, &kernelAxBpCpD, {SELECT_A,SELECT_B,SELECT_C,SELECT_B,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b + c + b", 15));

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | ((FUNCTION_MASK_C) << 3) | (FUNCTION_MASK_A << 6), Md, &kernelAxBpCpD, {SELECT_A,SELECT_B,SELECT_C,SELECT_A,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b + c + a", 16));

		nCoefficients = 3;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_C | (FUNCTION_MASK_B << 3), Md, &kernelAxBpC, {SELECT_A,SELECT_C,SELECT_B,0,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*c + b", 17));

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_C | ((FUNCTION_MASK_B) << 3) | (FUNCTION_MASK_A << 6), Md, &kernelAxBpCpD, {SELECT_A,SELECT_C,SELECT_B,SELECT_A,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*c + b + a", 18));

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_C | ((FUNCTION_MASK_B) << 3) | (FUNCTION_MASK_C << 6), Md, &kernelAxBpCpD, {SELECT_A,SELECT_C,SELECT_B,SELECT_C,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*c + b + c", 19));

		nCoefficients = 3;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_B | FUNCTION_MASK_C | (FUNCTION_MASK_A << 3), Md, &kernelAxBpC, {SELECT_B,SELECT_C,SELECT_A,0,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "b*c + a", 20));

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_B | FUNCTION_MASK_C | ((FUNCTION_MASK_A) << 3) | (FUNCTION_MASK_B << 6), Md, &kernelAxBpCpD, {SELECT_B,SELECT_C,SELECT_A,SELECT_B,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "b*c + a + b", 21));

		nCoefficients = 4;
		bf = calculateBestFunction(nCoefficients, nMeasurements, nThreads, nDimensions, b, FUNCTION_MASK_B | FUNCTION_MASK_C | ((FUNCTION_MASK_A) << 3) | (FUNCTION_MASK_C << 6), Md, &kernelAxBpCpD, {SELECT_B,SELECT_C,SELECT_A,SELECT_C,0,0});
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "b*c + a + c", 22));
	} else if (nDimensions == 4) {
		int nCoefficients = 2;
		char* bf = calculateBestFunction(nCoefficients, nMeasurements, batchSize, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | FUNCTION_MASK_D, Md, NULL, {0,0,0,0,0,0}, true, 0, &kernel4Dx);
		bestFunction = new Function(nCoefficients, bf, "a*b*c*d", 0);

		for (uint64_t batch = batchSize; batch < nThreads; batch += batchSize) {
			bf = calculateBestFunction(nCoefficients, nMeasurements, batchSize, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | FUNCTION_MASK_D, Md, NULL, {0,0,0,0,0,0}, true, batch, &kernel4Dx);
			bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c*d", 0));
		}

		nCoefficients = 5;
		bf = calculateBestFunction(nCoefficients, nMeasurements, batchSize, nDimensions, b, FUNCTION_MASK_A | (FUNCTION_MASK_B << 4) | (FUNCTION_MASK_C << 8) | (FUNCTION_MASK_D << 12), Md, NULL, {0,0,0,0,0,0}, true, 0, &kernel4Dp);
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a + b + c + d", 1));

		for (uint64_t batch = batchSize; batch < nThreads; batch += batchSize) {
			bf = calculateBestFunction(nCoefficients, nMeasurements, batchSize, nDimensions, b, FUNCTION_MASK_A | (FUNCTION_MASK_B << 4) | (FUNCTION_MASK_C << 8) | (FUNCTION_MASK_D << 12), Md, NULL, {0,0,0,0,0,0}, true, batch, &kernel4Dp);
			bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a + b + c + d", 1));
		}
	} else if (nDimensions == 5) {
		int nCoefficients = 2;
		char* bf = calculateBestFunction(nCoefficients, nMeasurements, batchSize, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | FUNCTION_MASK_D | FUNCTION_MASK_E, Md, NULL, {0,0,0,0,0,0}, true, 0, &kernel5Dx);
		bestFunction = new Function(nCoefficients, bf, "a*b*c*d*e", 0);

		for (uint64_t batch = batchSize; batch < nThreads; batch += batchSize) {
			bf = calculateBestFunction(nCoefficients, nMeasurements, batchSize, nDimensions, b, FUNCTION_MASK_A | FUNCTION_MASK_B | FUNCTION_MASK_C | FUNCTION_MASK_D | FUNCTION_MASK_E, Md, NULL, {0,0,0,0,0,0}, true, batch, &kernel5Dx);
			bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a*b*c*d*e", 0));
		}

		nCoefficients = 6;
		bf = calculateBestFunction(nCoefficients, nMeasurements, batchSize, nDimensions, b, FUNCTION_MASK_A | (FUNCTION_MASK_B << 5) | (FUNCTION_MASK_C << 10) | (FUNCTION_MASK_D << 15) | (FUNCTION_MASK_E << 20), Md, NULL, {0,0,0,0,0,0}, true, 0, &kernel5Dp);
		bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a + b + c + d + e", 1));

		for (uint64_t batch = batchSize; batch < nThreads; batch += batchSize) {
			bf = calculateBestFunction(nCoefficients, nMeasurements, batchSize, nDimensions, b, FUNCTION_MASK_A | (FUNCTION_MASK_B << 5) | (FUNCTION_MASK_C << 10) | (FUNCTION_MASK_D << 15) | (FUNCTION_MASK_E << 20), Md, NULL, {0,0,0,0,0,0}, true, batch, &kernel5Dp);
			bestFunction = getBestFunctionAndDestroyOther(bestFunction, new Function(nCoefficients, bf, "a + b + c + d + e", 1));
		}
	} else {
		printf("Currently unimplemented\n");
	}

	cudaFree(bd);
	cudaFree(pAd);
	cudaFree(rssSmape);
	cudaFree(coefD);
	cudaFree(lsMatricesD);
	free(b);
	matrix_free_gpu(Md);

	return bestFunction;
}
