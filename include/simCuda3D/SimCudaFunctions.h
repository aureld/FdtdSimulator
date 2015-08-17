//SimCudaFunctions.h: cuda simulation functions exported to the FDTD simulator 
//Aurelien Duval 2015
#pragma once
#ifndef _SIMCUDAFUNCTIONS_H_
#define _SIMCUDAFUNCTIONS_H_

#include <vector>
#include "SimCuda3D/Cuda_grid.h"

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



//creates a pointer in unified memory to the grid
grid *CudaInitGrid();

//initializes the host and device memory chunks for simulation
void CudaInitFields(grid *g);

//Initialize the E source auxiliary array
//srcField: 0-> Ex, 1->Ey, 2->Ez
void CudaInitializeSourceE(grid *g);

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
bool CudaCollectTimeSeriesData(__int64 timestep, int detX, int detY, int detZ, grid *g);

//writes the time series data to file
bool CudaWriteTimeSeriesData(char* filename, grid *g);

//Multiplex time series data for use in DS_Observer
std::vector< float > CudaMultiplexTimeSeriesData(unsigned __int8 fcomps, int nbTimeSteps);

#endif /*_SIMCUDAFUNCTIONS_H_*/
