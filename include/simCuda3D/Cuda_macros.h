//Cuda_macros.h: macros specific to CUDA  simulations
// Aurelien Duval 2015

#pragma once
#ifndef _CUDA_MACROS_H_
#define _CUDA_MACROS_H_

 
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include <Windows.h>

#define CDTDS 0.57735026918962576450914878050196 // 1/sqrt(3)

#define CUDA_ALLOC_3D(PNTR, NUMX, NUMY, NUMZ, TYPE)						\
    HANDLE_ERROR( cudaMalloc(&PNTR, (NUMX) * (NUMY) * (NUMZ) * sizeof(TYPE)));

#define CUDA_ALLOC_1D(PNTR, NUM, TYPE)						\
    HANDLE_ERROR( cudaMalloc(&PNTR, (NUM)* sizeof(TYPE)));

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

/*Following macros found in B-Calm implementation and stackexchange */

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#  define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_CHECK(call);                                                        \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
        } } while (0)




#endif /*_CUDA_MACROS_H_*/