// kernels.cu : update equations for E and H - Cuda enabled
// Aurelien Duval 2015
//see ACES J. 25(4) p303 (2010)

#include <stdio.h>
#include "cuda_runtime.h"
#include "SimCuda3D/cuda_macros.h"
#include "SimCuda3D/cuda_protos.h"
#include "SimCuda3D/Cuda_grid.h"
#include <math.h>

//array indexing macros
#define IDX(i, j, k) ((i) + (j) * (g->nx) + (k) * (g->nx) * (g->ny) )
#define K(index) (index / (g->nx * g->ny))
#define J(index) ((index - (K(index)*g->nx*g->ny))/g->nx)
#define I(index) ((index) - J(index) * g->nx - K(index) * g->nx * g->ny)



//initialize all device field arrays to zero before use (Seems there's no 
__global__ void Cuda_initFieldArrays( grid *g)
{
    //grid stride loop
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < g->domainSize; index += blockDim.x * gridDim.x)
    {
        g->ex[index] = 0.0;  
        g->ey[index] = 0.0; 
        g->ez[index] = 0.0; 
        g->hx[index] = 0.0; 
        g->hy[index] = 0.0; 
        g->hz[index] = 0.0; 
        g->Ca[index] = 0.0;
        g->Cb1[index] = 0.0;
        g->Cb2[index] = 0.0;
        g->Db1[index] = 0.0;
        g->Db2[index] = 0.0;
    }
}


//update equations for H fields - Cuda naive approach
__device__ void Cuda_updateH(grid *g)
{
    int i, j, k, pos;

    //grid stride loop
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < g->domainSize; index += blockDim.x * gridDim.x)
    {
        i = I(index);
        j = J(index);
        k = K(index);
        pos = IDX(i, j, k);

        if (i > g->nx-1  || j > g->ny-1 || k > g->nz-1) return;
        if (i <= 1 || j <= 1 || k <= 1) return;

        Cuda_updateHComponent(0, g->hx, i, j, k, pos, g);
        Cuda_updateHComponent(1, g->hy, i, j, k, pos, g);
        Cuda_updateHComponent(2, g->hz, i, j, k, pos, g);

    }

    return;
}

__device__ inline void Cuda_updateHComponent(int component, float *h, int i, int j, int k, int pos, grid *g)
{
    float e1a, e1b, e2a, e2b;
    switch (component)
    {
    case 0: //X
        e1a = g->ey[pos];
        e1b = g->ey[IDX(i, j, k - 1)];
        e2a = g->ez[pos];
        e2b = g->ez[IDX(i, j - 1, k)];
        break;
    case 1: //Y
        e1a = g->ez[pos];
        e1b = g->ez[IDX(i - 1, j, k)];
        e2a = g->ex[pos];
        e2b = g->ex[IDX(i, j, k - 1)];
        break;
    case 2: //Z
        e1a = g->ex[pos];
        e1b = g->ex[IDX(i, j - 1, k)];
        e2a = g->ey[pos];
        e2b = g->ey[IDX(i - 1, j, k)];
        break;
    }
    h[pos] = h[pos] + g->Db1[pos] * (e1a - e1b) - g->Db2[pos] * (e2a - e2b);
}


__device__ void Cuda_updateHBoundaries(grid *g)
{
    int i, j, k, pos;
    //grid stride loop
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < g->domainSize; index += blockDim.x * gridDim.x)
    {
        i = I(index);
        j = J(index);
        k = K(index);
        pos = IDX(i, j, k);

        if (i == 0)             //left
            Cuda_updateHBoundaryMinusX(i, j, k, pos, g);
        if (i == g->nx - 1)     //right
            Cuda_updateHBoundaryPlusX(i, j, k, pos, g);
        if (j == 0)             //bottom
            Cuda_updateHBoundaryMinusY(i, j, k, pos, g);
        if (j == g->ny - 1)     //top
            Cuda_updateHBoundaryPlusY(i, j, k, pos, g);
        if (k == 0)             //back
            Cuda_updateHBoundaryMinusZ(i, j, k, pos, g);
        if (k == g->nz - 1)     //front
            Cuda_updateHBoundaryPlusZ(i, j, k, pos, g);

    }

}

//update equations for E fields - Cuda naive approach
__device__ void Cuda_updateE(grid *g)
{
    int i, j, k, pos;

    //grid stride loop
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < g->domainSize; index += blockDim.x * gridDim.x)
    {
        i = I(index);
        j = J(index);
        k = K(index);
        pos = IDX(i, j, k);
        if (i > g->nx - 1 || j > g->ny - 1 || k > g->nz - 1) return;

        Cuda_updateEComponent(0, g->ex, i, j, k, pos, g);
        Cuda_updateEComponent(1, g->ey, i, j, k, pos, g);
        Cuda_updateEComponent(2, g->ez, i, j, k, pos, g);
    }
    return;
}

__device__ inline void Cuda_updateEComponent( int component, float *e, int i, int j, int k, int pos, grid *g)
{
    float h1a, h1b, h2a, h2b;
    switch (component)
    {
    case 0:
        h1a = g->hz[IDX(i, j + 1, k)];
        h1b = g->hz[pos];
        h2a = g->hy[IDX(i, j, k + 1)];
        h2b = g->hy[pos];
        break;
    case 1:
        h1a = g->hx[IDX(i, j, k + 1)];
        h1b = g->hx[pos];
        h2a = g->hz[IDX(i + 1, j, k)];
        h2b = g->hz[pos];
        break;
    case 2:
        h1a = g->hy[IDX(i + 1, j, k)];
        h1b = g->hy[pos];
        h2a = g->hx[IDX(i, j + 1, k)];
        h2b = g->hx[pos];
        break;
    }
    e[pos] = g->Ca[pos] * e[pos] + g->Cb1[pos] * (h1a - h1b) - g->Cb2[pos] * (h2a - h2b);
}

__device__ void Cuda_updateEBoundaries(grid *g)
{
    int i, j, k, pos;
    //grid stride loop
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < g->domainSize; index += blockDim.x * gridDim.x)
    {
        i = I(index);
        j = J(index);
        k = K(index);
        pos = IDX(i, j, k);

        if (i == 0)             //left
            Cuda_updateEBoundaryMinusX(i, j, k, pos, g);
        if (i == g->nx - 1)     //right
            Cuda_updateEBoundaryPlusX(i, j, k, pos, g);
        if (j == 0)             //bottom
            Cuda_updateEBoundaryMinusY(i, j, k, pos, g);
        if (j == g->ny - 1)     //top
            Cuda_updateEBoundaryPlusY(i, j, k, pos, g);
        if (k == 0)             //back
            Cuda_updateEBoundaryMinusZ(i, j, k, pos, g);
        if (k == g->nz - 1)     //front
            Cuda_updateEBoundaryPlusZ(i, j, k, pos, g);

    }

}


// update equations for E boundary fields -X side - Cuda naive approach
//PEC only for now
__device__ inline void Cuda_updateEBoundaryMinusX(int i, int j, int k, int pos, grid *g)
{
        if (j >= g->ny-1 || k >= g->nz-1) return;

        g->ex[pos] = 0.0; //PEC
        g->ey[pos] = 0.0;
        g->ez[pos] = 0.0;
}
 
// update equations for E boundary fields +X side - Cuda naive approach
//PEC only for now
__device__ inline void Cuda_updateEBoundaryPlusX(int i, int j, int k, int pos, grid *g)
{
        if (j >= g->ny-1 || k >= g->nz-1) return;

        g->ex[pos] = 0.0; //PEC
        g->ey[pos] = 0.0;
        g->ez[pos] = 0.0;
}

// update equations for E boundary fields -Y side - Cuda naive approach
//PEC only for now
__device__ inline void Cuda_updateEBoundaryMinusY(int i, int j, int k, int pos, grid *g)
{
        if (i >= g->nx-1 || k >= g->nz-1) return;

        g->ex[pos] = 0.0; //PEC
        g->ey[pos] = 0.0;
        g->ez[pos] = 0.0;
}

// update equations for E boundary fields +Y side - Cuda naive approach
//PEC only for now
__device__ inline void Cuda_updateEBoundaryPlusY(int i, int j, int k, int pos, grid *g)
{
        if (i >= g->nx-1 || k >= g->nz-1) return;

        g->ex[pos] = 0.0; //PEC
        g->ey[pos] = 0.0;
        g->ez[pos] = 0.0;
}

// update equations for E boundary fields -Z side - Cuda naive approach
//PEC only for now
__device__ inline void Cuda_updateEBoundaryMinusZ(int i, int j, int k, int pos, grid *g)
{
   if (i >= g->nx-1 || j >= g->ny-1) return;

    g->ex[pos] = 0.0; //PEC
    g->ey[pos] = 0.0;
    g->ez[pos] = 0.0;
}

// update equations for E boundary fields +Z side - Cuda naive approach
//PEC only for now
__device__ inline void Cuda_updateEBoundaryPlusZ(int i, int j, int k, int pos, grid *g)
{
    if (i >= g->nx-1 || j >= g->ny-1) return;

    g->ex[pos] = 0.0; //PEC
    g->ey[pos] = 0.0;
    g->ez[pos] = 0.0;
}


// update equations for H boundary fields -X side - Cuda naive approach
//PEC only for now
__device__ inline void Cuda_updateHBoundaryMinusX(int i, int j, int k, int pos, grid *g)
{
        if (j >= g->ny-1 || k >= g->nz-1) return;

        g->hx[pos] = 0.0; //PEC
        g->hy[pos] = 0.0;
        g->hz[pos] = 0.0;
}

// update equations for H boundary fields +X side - Cuda naive approach
//PEC only for now
__device__ inline void Cuda_updateHBoundaryPlusX(int i, int j, int k, int pos, grid *g)
{
        if (j >= g->ny-1 || k >= g->nz-1) return;

        g->hx[pos] = 0.0; //PEC
        g->hy[pos] = 0.0;
        g->hz[pos] = 0.0;
}

// update equations for H boundary fields -Y side - Cuda naive approach
//PEC only for now
__device__ inline void Cuda_updateHBoundaryMinusY(int i, int j, int k, int pos, grid *g)
{
    if (i >= g->nx-1 || k >= g->nz-1) return;

        g->hx[pos] = 0.0; //PEC
        g->hy[pos] = 0.0;
        g->hz[pos] = 0.0;
}

// update equations for H boundary fields +Y side - Cuda naive approach
//PEC only for now
__device__ inline void Cuda_updateHBoundaryPlusY(int i, int j, int k, int pos, grid *g)
{
        if (i >= g->nx-1 || k >= g->nz-1) return;

        g->hx[pos] = 0.0; //PEC
        g->hy[pos] = 0.0;
        g->hz[pos] = 0.0;
}


// update equations for H boundary fields -Z side - Cuda naive approach
//PEC only for now
__device__ inline void Cuda_updateHBoundaryMinusZ(int i, int j, int k, int pos, grid *g)
{
        if (i >= g->nx-1 || j >= g->ny-1) return;

        g->hx[pos] = 0.0; //PEC
        g->hy[pos] = 0.0;
        g->hz[pos] = 0.0;
}

// update equations for H boundary fields +Z side - Cuda naive approach
//PEC only for now
__device__ inline void Cuda_updateHBoundaryPlusZ(int i, int j, int k, int pos, grid *g)
{
      if (i >= g->nx-1 || j >= g->ny-1) return;

        g->hx[pos] = 0.0; //PEC
        g->hy[pos] = 0.0;
        g->hz[pos] = 0.0;
}




//Auxiliary source array initialization  
__global__ void Cuda_InitializeSrc(grid *g)
{
    //grid stride loop
    for (unsigned __int64 index = blockIdx.x * blockDim.x + threadIdx.x; index < g->nt; index += blockDim.x * gridDim.x)
    {
        if (index > g->nt) return;

        double d_efftime = index * g->dt;
        double envelope = 1.0 - exp(-(d_efftime / g->rTime)); //CW for now
        g->srcField[index] = g->amplitude * envelope * sin(g->omega * d_efftime + g->initPhase);
    }
    return;
}

//E field injection
//for single a point source, only 1 thread should be launched 
__device__ void Cuda_injectE(grid *g)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = I(index);
    int j = J(index);
    int k = K(index);
    int pos = IDX(i, j, k);

    //only 1 thread should update the field
    if (pos != g->srclinpos) return;

    float *field = NULL;
    switch (g->srcFieldComp)
    {
    case 0:
        field = g->ex;
        break;
    case 1:
        field = g->ey;
        break;
    case 2:
        field = g->ez;
        break;
    }
    field[g->srclinpos] += g->srcField[g->currentIteration];
}

//collect field data for select component and store in detector array for each timestep
__global__ void Cuda_CollectTimeSeriesData(float *component, float *field, int posx, int posy, int posz, __int64 timestep, grid *g)
{
    int pos = IDX(posx, posy, posz);
    component[timestep] = field[pos];
}


//whole step done on device
__global__ void Cuda_CalculateStep(grid *g)
{
    Cuda_updateE(g);
    Cuda_injectE(g);
    Cuda_updateEBoundaries(g);
    Cuda_updateH(g);
    Cuda_updateHBoundaries(g);

}
