#ifndef xitdrttnzc
#define xitdrttnzc

#include <stdlib.h>

struct CPUMatrix {
	int width;
	int height;
	float *elements;
};
struct GPUMatrix {
	int width;
	int height;
	size_t pitch; // row size in bytes
	float *elements;
};


CPUMatrix matrix_alloc_cpu(int width, int height);
CPUMatrix matrix_alloc_cpu_preallocatedElements(int width, int height, float* elements);
void matrix_free_cpu(CPUMatrix &m);

GPUMatrix matrix_alloc_gpu(int width, int height);
void matrix_free_gpu(GPUMatrix &m);

void matrix_upload(const CPUMatrix &src, GPUMatrix &dst);
void matrix_download(const GPUMatrix &src, CPUMatrix &dst);

#endif