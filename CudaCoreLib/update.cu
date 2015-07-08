// update.cu : update equations for E and H - Cuda enabled
// Aurelien Duval 2015
//see ACES J. 25(4) p303 (2010)

#include <stdio.h>
#include "cuda_runtime.h"
#include "Cuda_fdtd_macros.h"
#include "Cuda_fdtd-protos.h"
#include <math.h>

#define IDX(i, j, k) ((long)(i) * (ny) * (nz) + (long)(j) * (nz) + (long)(k))


//update equations for H fields - Cuda naive approach

__global__ void Cuda_updateHx(float *ex, float *cexe, float *cexh,
	float *ey, float *ceye, float *ceyh,
	float *ez, float *ceze, float *cezh,
	float *hx, float *chxe, float *chxh,
	float *hy, float *chye, float *chyh,
	float *hz, float *chze, float *chzh,
	const int nx, const int ny, const int nz,
	const int time)
{
	int i, j, k;

	i = blockIdx.x;
	j = blockIdx.y;
	k = threadIdx.x;


	if (i <= 0 || j <= 0 || k <= 0) return;
	if (i >= (nx) || j >= (ny-1) || k >= (nz-1)) return;

	int pos = IDX(i, j, k);


	hx[pos] = chxh[pos] * hx[pos]
		+ chxe[pos] * ((ey[IDX(i, j, k + 1)] - ey[pos])
		- (ez[IDX(i, j + 1, k)] - ez[pos]));

	return;

}

__global__ void Cuda_updateHy(float *ex, float *cexe, float *cexh,
	float *ey, float *ceye, float *ceyh,
	float *ez, float *ceze, float *cezh,
	float *hx, float *chxe, float *chxh,
	float *hy, float *chye, float *chyh,
	float *hz, float *chze, float *chzh,
	const int nx, const int ny, const int nz,
	const int time)
{
	int i, j, k;

	i = blockIdx.x;
	j = blockIdx.y;
	k = threadIdx.x;


	if (i <= 0 || j <= 0 || k <= 0) return;
	if (i >= (nx-1) || j >= (ny) || k >= (nz-1)) return;

	int pos = IDX(i, j, k);


	hy[pos] = chyh[pos] * hy[pos]
		+ chye[pos] * ((ez[IDX(i + 1, j, k)] - ez[pos])
		- (ex[IDX(i, j, k + 1)] - ex[pos]));

	return;

}

__global__ void Cuda_updateHz(float *ex, float *cexe, float *cexh,
	float *ey, float *ceye, float *ceyh,
	float *ez, float *ceze, float *cezh,
	float *hx, float *chxe, float *chxh,
	float *hy, float *chye, float *chyh,
	float *hz, float *chze, float *chzh,
	const int nx, const int ny, const int nz,
	const int time)
{
	int i, j, k;

	i = blockIdx.x;
	j = blockIdx.y;
	k = threadIdx.x;


	if (i <= 0 || j <= 0 || k <= 0) return;
	if (i >= (nx-1) || j >= (ny-1) || k >= (nz)) return;

	int pos = IDX(i, j, k);


	hz[pos] = chzh[pos] * hz[pos]
		+ chze[pos] * ((ex[IDX(i, j + 1, k)] - ex[pos])
		- (ey[IDX(i + 1, j, k)] - ey[pos]));

	return;
}

//update equations for E fields - Cuda naive approach


__global__ void Cuda_updateEx(float *ex, float *cexe, float *cexh,
	float *ey, float *ceye, float *ceyh,
	float *ez, float *ceze, float *cezh,
	float *hx, float *chxe, float *chxh,
	float *hy, float *chye, float *chyh,
	float *hz, float *chze, float *chzh,
	const int nx, const int ny, const int nz,
	const int time)
{
	int i, j, k;

	i = blockIdx.x;
	j = blockIdx.y;
	k = threadIdx.x;


	if (i <= 0 || j <= 1 || k <= 1) return;
	if (i >= (nx - 1) || j >= (ny - 1) || k >= (nz-1)) return;

	int pos = IDX(i, j, k);

	ex[pos] = cexe[pos] * ex[pos]
		+ cexh[pos] * ((hz[pos] - hz[IDX(i, j - 1, k)])
		- (hy[pos] - hy[IDX(i, j, k - 1)]));

	return;
}

__global__ void Cuda_updateEy(float *ex, float *cexe, float *cexh,
	float *ey, float *ceye, float *ceyh,
	float *ez, float *ceze, float *cezh,
	float *hx, float *chxe, float *chxh,
	float *hy, float *chye, float *chyh,
	float *hz, float *chze, float *chzh,
	const int nx, const int ny, const int nz,
	const int time)
{
	int i, j, k;

	i = blockIdx.x;
	j = blockIdx.y;
	k = threadIdx.x;


	if (i <= 1 || j <= 0 || k <= 1) return;
	if (i >= (nx - 1) || j >= (ny - 1) || k >= (nz-1)) return;

	int pos = IDX(i, j, k);

	ey[pos] = ceye[pos] * ey[pos]
		+ ceyh[pos] * ((hx[pos] - hx[IDX(i, j, k - 1)])
		- (hz[pos] - hz[IDX(i - 1, j, k)]));

	return;
}

__global__ void Cuda_updateEz(float *ex, float *cexe, float *cexh,
	float *ey, float *ceye, float *ceyh,
	float *ez, float *ceze, float *cezh,
	float *hx, float *chxe, float *chxh,
	float *hy, float *chye, float *chyh,
	float *hz, float *chze, float *chzh,
	const int nx, const int ny, const int nz,
	const int time)
{
	int i, j, k;

	i = blockIdx.x;
	j = blockIdx.y;
	k = threadIdx.x;


	if (i <= 1 || j <= 1 || k <= 0) return;
	if (i >= (nx - 1) || j >= (ny - 1) || k >= (nz-1)) return;

	int pos = IDX(i, j, k);

	ez[pos] = ceze[pos] * ez[pos]
		+ cezh[pos] * ((hy[pos] - hy[IDX(i - 1, j, k)])
		- (hx[pos] - hx[IDX(i, j - 1, k)]));

	return;
}




__global__ void Cuda_applySource(	float *ex, float *ey, float *ez, const int posX, const int posY, const int posZ, 
									const int nx, const int ny, const int nz, const double delay, const int time)
{
	//horribly wasteful since only 1 thread will go through this. 
	//But probably better than transferring to host memory and back
	ey[IDX(posX, posY, posZ)] += Cuda_ezInc(time, CDTDS, 10, 1.0);
	return;	
}


//calculates the input field at specified time and location
__device__ float Cuda_ezInc(const int time, const double cdtds, const int ppw, const double delay)
{
	double val;


	double coef = M_PI * M_PI * (cdtds * time / ppw - delay) *  (cdtds * time / ppw - delay);

	val = (1.0 - 2.0 * coef) * exp(-coef);

	return (float)val;

}