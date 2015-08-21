//Cuda_memory.h: functions related to memory manipulation
//Aurelien Duval 2015
#pragma once
#ifndef _CUDA_MEMORY_H_
#define _CUDA_MEMORY_H_

#include "simCuda3D/Cuda_grid.h"


void* AllocateAndCopyToDevice(void *h_data, unsigned int memsize);
void CopyToHost(void *d_data, void *h_data, unsigned int memsize);
bool CudaRetrieveAll(grid *g, grid *dg);
inline void CudaRetrieveField(float *h_data, float *d_data, unsigned long size);
//initializes the host and device memory chunks for simulation
bool CudaInitFields(grid *g, grid *dg);
//Allocate fields memory on the CUDA device
//nx, ny, nz refer to the main domain size (no boundaries)
void CudaAllocateFields(grid *g);
//free fields memory on the CUDA device
//nx, ny, nz refer to the main domain size (no boundaries)
void CudaFreeFields(grid *g);


#endif /*_CUDA_MEMORY_H_*/