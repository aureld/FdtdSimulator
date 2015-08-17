//Cuda_macros.h: macros specific to CUDA  simulations
// Aurelien Duval 2015

#pragma once
#ifndef _CUDA_MACROS_H_
#define _CUDA_MACROS_H_

 
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "windows.h"

#pragma warning(disable:4505)

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	char msg[1000];
	if (err != cudaSuccess) {
		sprintf(msg,"%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		OutputDebugString((LPCSTR)msg);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define CDTDS 0.57735026918962576450914878050196 // 1/sqrt(3)

#define CUDA_ALLOC_3D(PNTR, NUMX, NUMY, NUMZ, TYPE)						\
	HANDLE_ERROR( cudaMalloc(&PNTR, (NUMX) * (NUMY) * (NUMZ) * sizeof(TYPE)));

#define CUDA_ALLOC_MANAGED_3D(PNTR, NUMX, NUMY, NUMZ, TYPE)						\
	HANDLE_ERROR( cudaMallocManaged(&PNTR, (NUMX)* (NUMY)* (NUMZ)* sizeof(TYPE), cudaMemAttachHost)); 

#define CUDA_ALLOC_MANAGED_1D(PNTR, NUMEL, TYPE)						\
	HANDLE_ERROR( cudaMallocManaged(&PNTR, (NUMEL)* sizeof(TYPE), cudaMemAttachHost)); 

#define ALLOC_3D(PNTR, NUMX, NUMY, NUMZ, TYPE)						\
	PNTR = (TYPE *)calloc((NUMX) * (NUMY) * (NUMZ), sizeof(TYPE));	\
	if (!PNTR) {													\
		perror("ALLOC_3D");											\
		fprintf(stderr, "Allocation failed for " #PNTR" \n");		\
		exit(-1);													\
					}


#endif /*_CUDA_MACROS_H_*/