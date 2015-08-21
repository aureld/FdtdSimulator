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
void CudaGetBlockSize(unsigned int &blocksize, unsigned int &gridsize, grid *g);

//creates a pointer in device memory for the grid
bool CudaInitGrid(grid *g, grid *dg);

//do all calculations for the current timestep
bool CudaCalculateStep(int blocksize, int gridsize, grid *g, unsigned long iteration);

bool CudaWriteTimeSeriesData(char* filename, float *det_data, unsigned long nt);

void PrepareFrame(grid *g, unsigned char * buf);
double base(double val);
double red(double gray);
double green(double gray);
double blue(double gray);


/*************** device side functions ************************/


__device__ inline void Cuda_updateHComponent(int component, float *h, int i, int j, int k, int pos, float *ex, float *ey, float *ez, float *Db1, float *Db2, unsigned int *mat );
__device__ inline void Cuda_updateEComponent(int component, float *e, int i, int j, int k, int pos, float *hx, float *hy, float *hz, float *Ca, float *Cb1, float *Cb2, unsigned int *mat, float srcField);
__global__ void Cuda_updateH(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *Db1, float *Db2, unsigned int *mat);
__global__ void Cuda_updateE(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *Ca, float *Cb1, float *Cb2, unsigned int *mat, float *srcField, unsigned long iteration);
__global__ void Cuda_CaptureFields(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *detEx, unsigned long iteration);



#endif /*_SIMCUDAFUNCTIONS_H_*/
