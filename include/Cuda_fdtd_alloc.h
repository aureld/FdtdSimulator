//Cuda_fdtd_alloc.h: memory allocation of the fields etc - Cuda capable
// Aurelien Duval 2015
#pragma once 
#ifndef _CUDA_FDTD_ALLOC_H_
#define _CUDA_FDTD_ALLOC_H_

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "Cuda_fdtd_macros.h"


//Memory macros
#define CUDA_ALLOC_1D(PNTR, NUM, TYPE)									\
	HANDLE_ERROR( cudaMalloc(&PNTR, NUM *  sizeof(TYPE)));
	

#define CUDA_ALLOC_2D(PNTR, NUMX, NUMY, TYPE)							\
	HANDLE_ERROR( cudaMalloc(&PNTR, (NUMX) * (NUMY) *  sizeof(TYPE)));
			

#define CUDA_ALLOC_3D(PNTR, NUMX, NUMY, NUMZ, TYPE)						\
	HANDLE_ERROR( cudaMalloc(&PNTR, (NUMX) * (NUMY) * (NUMZ) * sizeof(TYPE)));



#endif /*_CUDA_FDTD_ALLOC_H_*/