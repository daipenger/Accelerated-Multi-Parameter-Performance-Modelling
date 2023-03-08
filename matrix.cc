#include <stdio.h>
#include <cuda_runtime.h>

#include "matrix.h"

CPUMatrix matrix_alloc_cpu(int width, int height)
{
	CPUMatrix m;
	m.width = width;
	m.height = height;
	m.elements = new float[m.width * m.height];
	return m;
}

CPUMatrix matrix_alloc_cpu_preallocatedElements(int width, int height, float* elements) {
	CPUMatrix m;
	m.width = width;
	m.height = height;
	m.elements = elements;
	return m;
}

void matrix_free_cpu(CPUMatrix &m)
{
	delete[] m.elements;
}

GPUMatrix matrix_alloc_gpu(int width, int height)
{
	GPUMatrix m;
	m.width = width;
	m.height = height;
	cudaMallocPitch((void**) &m.elements, &m.pitch, sizeof(float) * width, height);
	return m;
}

void matrix_free_gpu(GPUMatrix &m)
{
	cudaFree(m.elements);
}

void matrix_upload(const CPUMatrix &src, GPUMatrix &dst)
{
	if (src.width != dst.width || src.height != dst.height) {
		puts("matrix_upload: width or height does not match. Will not copy.\n");
		return;
	}

	cudaMemcpy2D(dst.elements, dst.pitch, src.elements, src.width * sizeof(float), src.width * sizeof(float), src.height, cudaMemcpyHostToDevice);
}

void matrix_download(const GPUMatrix &src, CPUMatrix &dst)
{
	if (src.width != dst.width || src.height != dst.height) {
		puts("matrix_download: width or height does not match. Will not copy.\n");
		return;
	}

	cudaMemcpy2D(dst.elements, dst.width * sizeof(float), src.elements, src.pitch, dst.width * sizeof(float), dst.height, cudaMemcpyDeviceToHost);
}
