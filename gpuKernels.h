#ifndef nsctuowldfarlh
#define nsctuowldfarlh

#include <stdint.h>
#include <stdio.h>
#include "matrix.h"

#define sizeI 13
#define sizeJ 3

extern const float I[sizeI];
extern const float J[sizeJ];

struct Function {
	int nCoefficients;
	// float^nCoefficients
	float* coefficients;
	float rssSmape;
	uint64_t exponents;
	// Description of function, e.g. "a*b"
	const char* functionDescription;
	// Magic number for the function description
	int functionType;

	Function(int nCoef, char* funcData, const char* funcDesc, int funcType)
	: nCoefficients(nCoef), functionDescription(funcDesc), functionType(funcType)
	{
		coefficients = (float*) funcData;
		rssSmape = coefficients[nCoef];
		exponents = *((uint64_t*)(&coefficients[nCoef+1]));
	}

	~Function() {
		free(coefficients);
	}

	float getError() {
		return rssSmape;
	}

	void print() {
		printf("%s\n", functionDescription);
		printf("Error: %f\n", getError());
		printf("Coefficients: ");
		for (int i = 0; i < nCoefficients; ++i) {
			printf("%f ", coefficients[i]);
		}
		printf("\nExponents: ");
		uint64_t functionId = exponents;
		for (int i = 1; functionId != 0; ++i) {
			float is = I[functionId % sizeI];
			functionId /= sizeI;
			float js = J[functionId % sizeJ];
			functionId /= sizeJ;
			printf("i%d=%f j%d=%f ", i, is, i, js);
		}
		puts("");
	}
};

// M contains the measurements.
// It is one row per measurement:
// Therefore M.width - 1 is the amount of parameters / the dimension of the function
// The last element in each a row is the measured runtime
// M.height is therefore the amount of measurements
Function* process(const CPUMatrix &M);

void duplicateData(const float* __restrict__ in, const int inLen, float* __restrict__ out, const int outLen);

#endif