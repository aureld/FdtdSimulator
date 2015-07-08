//Cuda_fdtd-protos.h: functions prototypes for CUDA enabled simulations
// Aurelien Duval 2015

#pragma once
#ifndef _CUDA_FDTD_PROTOS_H_
#define _CUDA_FDTD_PROTOS_H_


#include "fdtd-macros.h"
#include <stdlib.h>


//forward declarations
class Movie;

//prototypes
void Cuda_gridInit(Grid *,int, int, int, int);

__global__ 
void Cuda_updateHx(	float *ex, float *cexe, float *cexh,
					float *ey, float *ceye, float *ceyh,
					float *ez, float *ceze, float *cezh,
					float *hx, float *chxe, float *chxh,
					float *hy, float *chye, float *chyh,
					float *hz, float *chze, float *chzh,
					const int nx, const int ny, const int nz,
					const int time);

__global__
void Cuda_updateHy(	float *ex, float *cexe, float *cexh,
					float *ey, float *ceye, float *ceyh,
					float *ez, float *ceze, float *cezh,
					float *hx, float *chxe, float *chxh,
					float *hy, float *chye, float *chyh,
					float *hz, float *chze, float *chzh,
					const int nx, const int ny, const int nz,
					const int time);

__global__
void Cuda_updateHz(	float *ex, float *cexe, float *cexh,
					float *ey, float *ceye, float *ceyh,
					float *ez, float *ceze, float *cezh,
					float *hx, float *chxe, float *chxh,
					float *hy, float *chye, float *chyh,
					float *hz, float *chze, float *chzh,
					const int nx, const int ny, const int nz,
					const int time);

__global__ 
void Cuda_updateEx(	float *ex, float *cexe, float *cexh,
					float *ey, float *ceye, float *ceyh,
					float *ez, float *ceze, float *cezh,
					float *hx, float *chxe, float *chxh,
					float *hy, float *chye, float *chyh,
					float *hz, float *chze, float *chzh,
					const int nx, const int ny, const int nz,
					const int time);

__global__
void Cuda_updateEy(	float *ex, float *cexe, float *cexh,
					float *ey, float *ceye, float *ceyh,
					float *ez, float *ceze, float *cezh,
					float *hx, float *chxe, float *chxh,
					float *hy, float *chye, float *chyh,
					float *hz, float *chze, float *chzh,
					const int nx, const int ny, const int nz,
					const int time);

__global__
void Cuda_updateEz(	float *ex, float *cexe, float *cexh,
					float *ey, float *ceye, float *ceyh,
					float *ez, float *ceze, float *cezh,
					float *hx, float *chxe, float *chxh,
					float *hy, float *chye, float *chyh,
					float *hz, float *chze, float *chzh,
					const int nx, const int ny, const int nz,
					const int time);

__global__ 
void Cuda_applySource(		float *ex, float *ey, float *ez, 
							const int posX, const int posY, const int posZ, 
							const int nx, const int ny, const int nz,
							const double delay, const int time);




__device__ float Cuda_ezInc(const int time, const double cdtds, const int ppw = 10, const double delay = 0.0);


void Cuda_do_time_stepping(Grid *g, Movie *movie);



#endif /*_CUDA_FDTD_PROTOS_H_*/
