//Cuda_protos.h: functions prototypes for CUDA enabled simulations
// Aurelien Duval 2015

#pragma once
#ifndef _CUDA_PROTOS_H_
#define _CUDA_PROTOS_H_

#include "SimCuda3D/Cuda_grid.h"

//kernels 

__global__ void Cuda_CalculateStep(grid *g);
__global__ void Cuda_initFieldArrays(grid *g);
//__global__ void Cuda_InitializeSrc(grid *g);


//__global__ void Cuda_CollectTimeSeriesData(float *component, float *field, int posx, int posy, int posz, __int64 timestep, grid *g);


//device functions
__device__ void Cuda_updateE(grid *g);
__device__ void Cuda_updateEBoundaries(grid *g);
__device__ void Cuda_updateH(grid *g);
__device__ void Cuda_updateHBoundaries(grid *g);

__device__ void Cuda_injectE(grid *g);

__device__ inline void Cuda_updateEComponent(int component, float *e, int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateHComponent(int component, float *h, int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateEBoundaryMinusX(int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateEBoundaryPlusX(int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateEBoundaryMinusY(int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateEBoundaryPlusY(int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateEBoundaryMinusZ(int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateEBoundaryPlusZ(int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateHBoundaryMinusX(int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateHBoundaryPlusX(int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateHBoundaryMinusY(int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateHBoundaryPlusY(int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateHBoundaryMinusZ(int i, int j, int k, int pos, grid *g);
__device__ inline void Cuda_updateHBoundaryPlusZ(int i, int j, int k, int pos, grid *g);

#endif /*_CUDA_PROTOS_H_*/
