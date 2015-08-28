//Cuda_constantMemoryData.h: CUDA constant memory declarations (must be separated since only file scope allowed)
// Aurelien Duval 2015

#pragma once
#ifndef _CUDA_CONSTANTMEMORYDATA_H_
#define _CUDA_CONSTANTMEMORYDATA_H_

#include <cuda_runtime.h>
#include "simCuda3D/Cuda_grid.h"

//constant memory declarations 
__constant__ unsigned int NX;
__constant__ unsigned int NY;
__constant__ unsigned int NZ;
__constant__ unsigned int DOMAINSIZE;
__constant__ int    SRCLINPOS;
__constant__ int    SRCFIELDCOMP;
__constant__ float DT;
__constant__ float *EX;
__constant__ float *EY;
__constant__ float *EZ;
__constant__ float *HX;
__constant__ float *HY;
__constant__ float *HZ;
__constant__ float *DETEX;
__constant__ unsigned int *MAT;
__constant__ float EPSILON[MAX_SIM_SIZE];
__constant__ float DEX[MAX_SIM_SIZE];
__constant__ float DEY[MAX_SIM_SIZE];
__constant__ float DEZ[MAX_SIM_SIZE];
__constant__ float DHX[MAX_SIM_SIZE];
__constant__ float DHY[MAX_SIM_SIZE];
__constant__ float DHZ[MAX_SIM_SIZE];
__constant__ float C1[MAX_NB_MAT];
__constant__ float C2[MAX_NB_MAT];

#endif /*_CUDA_CONSTANTMEMORYDATA_H_*/