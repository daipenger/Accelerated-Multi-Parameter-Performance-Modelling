#include "PARAMETERS.h"
#include <stdio.h>
#include <stdlib.h>
#include "gpuKernels.h"
#include "matrix.h"
#include "math.h"
#include <cuda_runtime.h>

// BEGIN OF TESTBENCH

void print_cuda_devices()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (int i = 0; i < deviceCount; ++i) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Multiprocessor count: %u\n", prop.multiProcessorCount);
		printf("Clock rate: %.3fGHz\n", prop.clockRate * 1e-6f);
		printf("Global memory: %luMiB\n", prop.totalGlobalMem / 1048576);
		printf("L2 cache size: %uKiB\n", prop.l2CacheSize / 1024);
	}
}

float getRandomFloat(float max) {
	return (float)rand()/(float)(RAND_MAX/max);
}

float testFunction1(float a, float b) {
	return 3.5 * a * pow(b, 2.5) + 5.4 * a + 12 + getRandomFloat(1);
}

float testFunction2(float a, float b, float c) {
	return a*b*c + a*b + c;
}

float testFunction3(float a, float b, float c, float d) {
	return 5 * a*b*c*d + 12;
}

float testFunction4(float a, float b, float c, float d, float e) {
	return 3*a + 4*b + 2*c + 1.8*d + e + 10;
}

int main() {
	printf("Accelerated-Multi-Parameter-Performance-Modelling testbench\n");
	#ifdef DRDEBUG
		printf("Debugging mode on!\n");
		print_cuda_devices();
	#else
		printf("Debugging mode off!\n");
	#endif
	cudaSetDevice(0);

	/*printf("Testing 2 dimensions\n");

	// 2 (See gpuKernels.h) Dimensions, 124 measured points
	CPUMatrix matrix = matrix_alloc_cpu(3, 124);
	for (int i = 0; i < 124; ++i) {
		float x = getRandomFloat(10);
		float y = getRandomFloat(10);
		matrix.elements[i*3] = x;
		matrix.elements[i*3+1] = y;
		matrix.elements[i*3+2] = testFunction1(x,y);
	}
	Function* f = process(matrix);
	printf("Found the best function:\n");
	f->print();
	delete f;
	matrix_free_cpu(matrix);*/

	printf("Testing 3 dimensions\n");

	CPUMatrix matrix = matrix_alloc_cpu(4, 125);
	for (int i = 0; i < 125; ++i) {
		float x = getRandomFloat(10);
		float y = getRandomFloat(10);
		float z = getRandomFloat(10);
		matrix.elements[i*4] = x;
		matrix.elements[i*4+1] = y;
		matrix.elements[i*4+2] = z;
		matrix.elements[i*4+3] = testFunction2(x,y,z);
	}
	Function* f = process(matrix);
	printf("Found the best function:\n");
	f->print();
	delete f;
	matrix_free_cpu(matrix);

	/*printf("Testing 4 dimensions\n");

	CPUMatrix matrix = matrix_alloc_cpu(5, 125);
	for (int i = 0; i < 125; ++i) {
		float x = getRandomFloat(10);
		float y = getRandomFloat(10);
		float z = getRandomFloat(10);
		float u = getRandomFloat(10);
		matrix.elements[i*5] = x;
		matrix.elements[i*5+1] = y;
		matrix.elements[i*5+2] = z;
		matrix.elements[i*5+3] = u;
		matrix.elements[i*5+4] = testFunction3(x,y,z,u);
	}
	Function* f = process(matrix);
	printf("Found the best function:\n");
	f->print();
	delete f;
	matrix_free_cpu(matrix);*/

	/*printf("Testing 5 dimensions\n");

	CPUMatrix matrix = matrix_alloc_cpu(6, 125);
	for (int i = 0; i < 125; ++i) {
		float x = getRandomFloat(10);
		float y = getRandomFloat(10);
		float z = getRandomFloat(10);
		float u = getRandomFloat(10);
		float v = getRandomFloat(10);
		matrix.elements[i*6] = x;
		matrix.elements[i*6+1] = y;
		matrix.elements[i*6+2] = z;
		matrix.elements[i*6+3] = u;
		matrix.elements[i*6+4] = v;
		matrix.elements[i*6+5] = testFunction4(x,y,z,u,v);
	}
	Function* f = process(matrix);
	printf("Found the best function:\n");
	f->print();
	delete f;
	matrix_free_cpu(matrix);*/

	printf("PMPP project testbench done\n");
	return 0;
}
