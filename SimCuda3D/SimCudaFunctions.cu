#include "SimCuda3D\Cuda_macros.h"
#include "SimCuda3D\SimCudaFunctions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SimCuda3D\Cuda_grid.h"
#include <vector>
#include <assert.h>
#include <stdlib.h>

cudaStream_t Stream;
dim3 GridSize;
dim3 BlockSize;


//constant memory variables 
__constant__ unsigned int NX;
__constant__ unsigned int NY;
__constant__ unsigned int NZ;
__constant__ unsigned int DOMAINSIZE;
__constant__ int    SRCLINPOS;
__constant__ int    SRCFIELDCOMP;



//array indexing macros
#define IDX(i, j, k) ((i) + (j) * (NX) + (k) * (NX) * (NY) )
#define K(index) (index / (NX * NY))
#define J(index) ((index - (K(index)*NX*NY))/NX)
#define I(index) ((index) - J(index) * NX - K(index) * NX * NY)


//allocates a chunck of global memory on device and copy the data there
void* AllocateAndCopyToDevice(void *h_data, unsigned int memsize) {
    // allocate device global memory
    void *d_data;
    CUDA_SAFE_CALL(cudaMalloc((void**)& d_data, memsize));
    // copy to device
    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, memsize, cudaMemcpyHostToDevice));
    return d_data;
}

//update equations for H fields
__global__ void Cuda_updateH(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *Db1, float *Db2)
{
    int i, j, k, pos;

    //grid stride loop
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < DOMAINSIZE; index += blockDim.x * gridDim.x)
    {
        i = I(index);
        j = J(index);
        k = K(index);
        pos = IDX(i, j, k);

        if (i > NX - 1 || j > NY - 1 || k > NZ - 1) return;
        if (i <= 1 || j <= 1 || k <= 1) return;

        Cuda_updateHComponent(0, hx, i, j, k, pos, ex, ey, ez, Db1, Db2);
        Cuda_updateHComponent(1, hy, i, j, k, pos, ex, ey, ez, Db1, Db2);
        Cuda_updateHComponent(2, hz, i, j, k, pos, ex, ey, ez, Db1, Db2);
    }
    return;
}

//H field update equations helper
__device__ inline void Cuda_updateHComponent(int component, float *h, int i, int j, int k, int pos, float *ex, float *ey, float *ez, float *Db1, float *Db2)
{
    float e1a, e1b, e2a, e2b;
    switch (component)
    {
    case 0: //X
        e1a = ey[pos];
        e1b = ey[IDX(i, j, k - 1)];
        e2a = ez[pos];
        e2b = ez[IDX(i, j - 1, k)];
        break;
    case 1: //Y
        e1a = ez[pos];
        e1b = ez[IDX(i - 1, j, k)];
        e2a = ex[pos];
        e2b = ex[IDX(i, j, k - 1)];
        break;
    case 2: //Z
        e1a = ex[pos];
        e1b = ex[IDX(i, j - 1, k)];
        e2a = ey[pos];
        e2b = ey[IDX(i - 1, j, k)];
        break;
    }
    h[pos] = h[pos] + Db1[pos] * (e1a - e1b) - Db2[pos] * (e2a - e2b);
}

/*

//whole step done on device
__global__ void Cuda_CalculateStep(grid *g)
{
   // Cuda_updateE(g);
   // Cuda_injectE(g);
   // Cuda_updateEBoundaries(g);
    Cuda_updateH(g);
   // Cuda_updateHBoundaries(g);

}
*/


//allocates the grid struct in device memory 
bool CudaInitGrid(grid *g, grid * dg)
{
    if (g == NULL)
    {
        perror("[CudaInitGrid]: host grid must be initialized");
        return false;
    }

    if (g->Ca == NULL || g->Cb1 == NULL || g->Cb2 == NULL || g->Db1 == NULL || g->Db2 == NULL)
    {
        perror("[CudaInitGrid]: host coefficient arrays must be initialized");
        return false;
    }
    
    if (g->srcField == NULL)
    {
        perror("[CudaInitGrid]: host source array must be initialized");
        return false;
    }

    //allocate and copy coefficients and source in global memory 
    //(they are constant for the sim, but allocated at runtime so still global mem)
    dg->Ca = (float*)AllocateAndCopyToDevice(g->Ca, sizeof(float)*g->domainSize);
    dg->Cb1 = (float*)AllocateAndCopyToDevice(g->Cb1, sizeof(float)*g->domainSize);
    dg->Cb2 = (float*)AllocateAndCopyToDevice(g->Cb2, sizeof(float)*g->domainSize);
    dg->Db1 = (float*)AllocateAndCopyToDevice(g->Db1, sizeof(float)*g->domainSize);
    dg->Db2 = (float*)AllocateAndCopyToDevice(g->Db2, sizeof(float)*g->domainSize);
    dg->srcField = (float*)AllocateAndCopyToDevice(g->srcField, sizeof(float)*g->nt);

    //allocate and copy constants to constant memory (limit: 64KB)

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NX, &(g->nx), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NY, &(g->ny), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NZ, &(g->nz), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(DOMAINSIZE, &(g->domainSize), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(SRCLINPOS, &(g->srclinpos), sizeof(int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(SRCFIELDCOMP, &(g->srcFieldComp), sizeof(int), 0, cudaMemcpyHostToDevice));
    return true;
}

//allocates the device fields in global memory
bool CudaInitFields(grid *g, grid *dg)
{
    if (g->ex == NULL || g->ey == NULL || g->ez == NULL || g->hx == NULL || g->hy == NULL || g->hz == NULL)
    {
        perror("[CudaInitGrid]: host fields must be initialized");
        return false;
    }
     
    dg->ex = (float*) AllocateAndCopyToDevice(g->ex, sizeof(float)*g->domainSize);
    dg->ey = (float*) AllocateAndCopyToDevice(g->ey, sizeof(float)*g->domainSize);
    dg->ez = (float*) AllocateAndCopyToDevice(g->ez, sizeof(float)*g->domainSize);
    dg->hx = (float*) AllocateAndCopyToDevice(g->hx, sizeof(float)*g->domainSize);
    dg->hy = (float*) AllocateAndCopyToDevice(g->hy, sizeof(float)*g->domainSize);
    dg->hz = (float*) AllocateAndCopyToDevice(g->hz, sizeof(float)*g->domainSize);

    return true;

}


//free memory on device for all the arrays
void CudaFreeFields(grid *g)
{
    cudaFree(g->ex);     cudaFree(g->ey);     cudaFree(g->ez); 
    cudaFree(g->hx);	 cudaFree(g->hy);     cudaFree(g->hz);	
    cudaFree(g->Ca);     cudaFree(g->Cb1);    cudaFree(g->Cb2);
    cudaFree(g->Db1);    cudaFree(g->Db2);
    cudaFree(g->srcField); 
    cudaFree(g->detEx);  cudaFree(g->detEy);  cudaFree(g->detEz);
    cudaFree(g->detHx);  cudaFree(g->detHy);  cudaFree(g->detHz);
    cudaFree(g);
    return;

}

//do all calculations for the current timestep
bool CudaCalculateStep(grid *g)
{
    Cuda_updateH << < 1, 1 >> >(g->ex, g->ey, g->ez, g->hx, g->hy, g->hz, g->Db1, g->Db2);
    CUDA_CHECK(cudaPeekAtLastError());
    return true;
}
