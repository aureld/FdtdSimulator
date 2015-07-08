//Cuda_fdtd_macros.h: macros specific to CUDA  simulations
// Aurelien Duval 2015

#pragma once
#ifndef _CUDA_FDTD_MACROS_H_
#define _CUDA_FDTD_MACROS_H_

 
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define CDTDS 0.57735026918962576450914878050196 // 1/sqrt(3)

#endif /*_CUDA_FDTD_MACROS_H_*/