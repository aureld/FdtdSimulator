//SimCudaFunctions.h: cuda simulation functions exported to the FDTD simulator 
//Aurelien Duval 2015
#pragma once
#ifndef _SIMCUDAFUNCTIONS_H_
#define _SIMCUDAFUNCTIONS_H_

#include <vector>
#include "SimCuda3D/Cuda_grid.h"
#include "cuda_runtime.h"

#pragma warning(disable : 4201)

//to code and retrieve field components in detectors
//for gcc, use "-fms-extensions" or name the struct to avoid errors
union FieldComps
{
    struct {
        unsigned __int8 Ex : 1;
        unsigned __int8 Ey : 1;
        unsigned __int8 Ez : 1;
        unsigned __int8 Hx : 1;
        unsigned __int8 Hy : 1;
        unsigned __int8 Hz : 1;
    };
    unsigned __int8 comps;
};

void* AllocateAndCopyToDevice(void *h_data, unsigned int memsize);

__global__ void Cuda_updateH(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *Db1, float *Db2);
__device__ inline void Cuda_updateHComponent(int component, float *h, int i, int j, int k, int pos, float *ex, float *ey, float *ez, float *Db1,float *Db2);



//creates a pointer in device memory for the grid
bool CudaInitGrid(grid *g, grid *dg);

//initializes the host and device memory chunks for simulation
bool CudaInitFields(grid *g, grid *dg);

//Initialize the E source auxiliary array
//srcField: 0-> Ex, 1->Ey, 2->Ez
//void CudaInitializeSourceE(grid *g);

//initialization of the point detector time series data
//nbTimeSteps: total number of timesteps
//detComps: field components selected (see FieldComps union)
void CudaInitDetectors( grid *g);

//Allocate fields memory on the CUDA device
//nx, ny, nz refer to the main domain size (no boundaries)
void CudaAllocateFields(grid *g);

//free fields memory on the CUDA device
//nx, ny, nz refer to the main domain size (no boundaries)
void CudaFreeFields(grid *g);


//do all calculations for the current timestep
bool CudaCalculateStep(grid *g);


//Collect time series data for the current point detector
//bool CudaCollectTimeSeriesData(__int64 timestep, int detX, int detY, int detZ, grid *g);

//writes the time series data to file
bool CudaWriteTimeSeriesData(char* filename, grid *g);

//Multiplex time series data for use in DS_Observer
std::vector< float > CudaMultiplexTimeSeriesData(unsigned __int8 fcomps, int nbTimeSteps);

#endif /*_SIMCUDAFUNCTIONS_H_*/
