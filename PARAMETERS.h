#pragma once
#include <stdbool.h>

//#define DRDEBUG

// If set to false: RSS is used instead
extern bool USE_SMAPE;
// WARNING: Leave-one-out cross-validation is very performance intensive!
extern bool USE_CROSSVALIDATION;