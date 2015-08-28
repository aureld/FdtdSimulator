//Cuda_memory.h: functions related to memory manipulation
//Aurelien Duval 2015
#pragma once
#ifndef _CUDA_MEMORY_H_
#define _CUDA_MEMORY_H_

#include "simCuda3D/Cuda_grid.h"
#include <cuda_runtime.h>

//array indexing macros
#define IDX(i, j, k) ((i) + (j) * (NX) + (k) * (NX) * (NY) )
#define K(index) (index / (NX * NY))
#define J(index) ((index - (K(index)*NX*NY))/NX)
#define I(index) ((index) - J(index) * NX - K(index) * NX * NY)


inline void CudaInitPinnedHost(void *ptr, size_t size);

void* AllocateAndCopyToDevice(void *h_data, size_t memsize);
void CopyToHost(void *d_data, void *h_data, unsigned int memsize);
void CopyToHost_Async(void *d_data, void *h_data, unsigned int memsize, cudaStream_t stream);

bool CudaRetrieveAll(grid *g, grid *dg);
void CudaRetrieveField(float *h_data, float *d_data, unsigned long size);
void CudaRetrieveField_Async(float *h_data, float *d_data, unsigned long size, cudaStream_t stream);


//initializes the host and device memory chunks for simulation
bool CudaInitGrid(grid *g, grid *dg);

void InitGridDeltas(grid *g, grid *dg);

void InitMaterialCoefs(grid *g, grid * dg);


//free fields memory on the CUDA device
//nx, ny, nz refer to the main domain size (no boundaries)
void CudaFreeFields(grid *g);


#endif /*_CUDA_MEMORY_H_*/