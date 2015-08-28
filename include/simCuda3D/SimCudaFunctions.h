//SimCudaFunctions.h: cuda simulation functions exported to the FDTD simulator 
//Aurelien Duval 2015
#pragma once
#ifndef _SIMCUDAFUNCTIONS_H_
#define _SIMCUDAFUNCTIONS_H_

#include <vector>
#include "SimCuda3D/Cuda_grid.h"
#include "cuda_runtime.h"




/*************** host side functions ***************************/

//returns the block and grid sizes for the kernel launches
void CudaGetBlockSize(dim3 &blocksize, dim3 &gridsize, grid *g);


//do all calculations for the current timestep
bool CudaCalculateStep(dim3 blocksize, dim3 gridsize, grid *g, unsigned long iteration, cudaStream_t stream);

bool CudaWriteTimeSeriesData(char* filename, float *det_data, unsigned long nt);

void PrepareFrame(grid *g, unsigned char * buf);
double base(double val);
double red(double gray);
double green(double gray);
double blue(double gray);


/*************** device side functions ************************/
//Kernels
__global__ void Cuda_updateH(unsigned long iteration);
__global__ void Cuda_updateE(unsigned long iteration);
__global__ void Cuda_CaptureFields(unsigned long iteration);




//helper "private" functions
__device__ inline void Cuda_updateHComponent(   int component, float *h, int i, int j, int k,
                                                float Ex_a[TILEXX + 1][TILEYY + 1], float Ex_b[TILEXX + 1][TILEYY + 1],
                                                float Ey_a[TILEXX + 1][TILEYY + 1], float Ey_b[TILEXX + 1][TILEYY + 1],
                                                float Ez_a[TILEXX + 1][TILEYY + 1], unsigned int mat);
__device__ inline void Cuda_updateEComponent(   int component, float *e, int pos,
                                                float Hx_a[TILEXX + 1][TILEYY + 1], float Hx_b[TILEXX + 1][TILEYY + 1],
                                                float Hy_a[TILEXX + 1][TILEYY + 1], float Hy_b[TILEXX + 1][TILEYY + 1],
                                                float Hz_a[TILEXX + 1][TILEYY + 1],
                                                float *hx, float *hy, float *hz, unsigned int *mat, float srcfield);
__device__ inline void Cuda_updateEComponent(int component, float *e, int i, int j, int k, int pos, float *hx, float *hy, float *hz, unsigned int *mat, float srcField);
__device__ void GetEShared(float E[TILEXX + 1][TILEXX + 1], int i, int j, int k, float *field);
__device__ void getHshared(float Hfield[TILEXX + 1][TILEYY + 1], int i, int j, int k, float *field);
__device__ void zeroShared(float A[TILEXX + 1][TILEYY + 1]);
__device__ void SwapShared_H(float A[TILEXX + 1][TILEYY + 1], float B[TILEXX + 1][TILEYY + 1]);
__device__ void SwapShared_E(float A[TILEXX + 1][TILEYY + 1], float B[TILEXX + 1][TILEYY + 1]);


#endif /*_SIMCUDAFUNCTIONS_H_*/
