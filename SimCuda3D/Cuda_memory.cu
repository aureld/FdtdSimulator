#include "simCuda3D/Cuda_memory.h"
#include "simCuda3D/Cuda_macros.h"
#include "simCuda3D/Cuda_constantMemoryData.h"
#include "common_defs.h"
#include <Windows.h>
#include <cuda_runtime.h>

//allocates a chunck of global memory on device and copy the data there
void* AllocateAndCopyToDevice(void *h_data, size_t memsize)
{
    // allocate device global memory
    void *d_data;
    CUDA_SAFE_CALL(cudaMalloc((void**)& d_data, memsize));
    // copy to device
    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, memsize, cudaMemcpyHostToDevice));
    return d_data;
}

//copies a chunk of global memory from device to host memory, async version
void CopyToHost_Async(void *d_data, void *h_data, unsigned int memsize, cudaStream_t stream)
{
    CUDA_SAFE_CALL(cudaMemcpyAsync(h_data, d_data, memsize, cudaMemcpyDeviceToHost, stream));
}

// copies a chunk of global memory from device to host memory
void CopyToHost(void *d_data, void *h_data, unsigned int memsize)
{
    CUDA_SAFE_CALL(cudaMemcpy(h_data, d_data, memsize, cudaMemcpyDeviceToHost));
}


inline void CudaInitPinnedHost(void *ptr, size_t size) {
    CUDA_SAFE_CALL(cudaMallocHost(&ptr,size)); // allocate a block of pinned memory (non-paged)
    memset(ptr, 0, size);
}

//init and allocate the grids
bool CudaInitGrid(grid *g, grid *dg)
{

    if (g == NULL)
    {
        perror("[CudaInitGrid]: host grid must be initialized");
        return false;
    }

    printf("Initializing the grid...");

    //init and allocated field on host and device
    size_t size = sizeof(float)*g->nt;

    CudaInitPinnedHost(g->detEx, size); //initiliazes a chunk of pinned host memory for the field data (enables async transfer)
    dg->detEx = (float*)AllocateAndCopyToDevice(g->detEx, size); //copy host data to device
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(DETEX, dg->detEx, sizeof(float*), 0, cudaMemcpyHostToDevice)); //store the device mem pointer in cst memory for easy access

    size = sizeof(float)*g->domainSize;

    CudaInitPinnedHost(g->ex, size);
    dg->ex = (float*)AllocateAndCopyToDevice(g->ex, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(EX, dg->ex, sizeof(float*), 0, cudaMemcpyHostToDevice));
    CudaInitPinnedHost(g->ey, size);
    dg->ey = (float*)AllocateAndCopyToDevice(g->ey, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(EY, dg->ey, sizeof(float*), 0, cudaMemcpyHostToDevice));
    CudaInitPinnedHost(g->ez, size);
    dg->ez = (float*)AllocateAndCopyToDevice(g->ez, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(EZ, dg->ez, sizeof(float*), 0, cudaMemcpyHostToDevice));
    CudaInitPinnedHost(g->hx, size);
    dg->hx = (float*)AllocateAndCopyToDevice(g->hx, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(HX, dg->hx, sizeof(float*), 0, cudaMemcpyHostToDevice));
    CudaInitPinnedHost(g->hy, size);
    dg->hy = (float*)AllocateAndCopyToDevice(g->hy, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(HY, dg->hy, sizeof(float*), 0, cudaMemcpyHostToDevice));
    CudaInitPinnedHost(g->hz, size);
    dg->hz = (float*)AllocateAndCopyToDevice(g->hz, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(HZ, dg->hz, sizeof(float*), 0, cudaMemcpyHostToDevice));


    //allocate and copy constants to constant memory (limit: 48KB / SM)
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NX, &(g->nx), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NY, &(g->ny), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NZ, &(g->nz), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(DT, &(g->dt), sizeof(float), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(DOMAINSIZE, &(g->domainSize), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    g->srclinpos = ((g->srcposX) + (g->srcposY)* (g->nx) + (g->srcposZ)* (g->nx)* (g->ny));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(SRCLINPOS, &(g->srclinpos), sizeof(int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(SRCFIELDCOMP, &(g->srcFieldComp), sizeof(int), 0, cudaMemcpyHostToDevice));


    if (g->mat == NULL)
    {
        perror("[CudaInitGrid]: host material grid must be initialized");
        return false;
    }
    dg->mat = (unsigned int*)AllocateAndCopyToDevice(g->mat, sizeof(unsigned int)*g->domainSize);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MAT, dg->mat, sizeof(unsigned int*), 0, cudaMemcpyHostToDevice));

    if (g->epsilon == NULL)
    {
        perror("[CudaInitGrid]: host epsilon grid must be initialized");
        return false;
    }

    //here we copy the values and not only pointers since the size is likely to be small. 
    size = sizeof(float)*g->Nmats;
    dg->epsilon = (float*)AllocateAndCopyToDevice(g->epsilon, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(EPSILON, &(dg->epsilon), size, 0, cudaMemcpyHostToDevice));

    printf("Done!\n");
    return true;
}


void InitMaterialCoefs(grid *g, grid * dg)
{
    printf("Initializing material coefficients...");

    g->C1 = (float *)cust_alloc(sizeof(float) * g->Nmats);
    g->C2 = (float *)cust_alloc(sizeof(float) * g->Nmats);

    for (int i = 0; i < g->Nmats; i++)
    {
        float sigma = 1.0; //dielectrics
        float coef = sigma*g->dt / (2 * g->epsilon[i]);
        g->C1[i] = (1 - coef) / (1 + coef);  // Ca in Tavlove
        g->C2[i] = (g->dt / (g->epsilon[i] * EPSILON_0)) / (1 + coef); //Cb1,2 in Tavlove (do we need eps0??)
    }

    size_t size = sizeof(float)*g->Nmats;
    dg->C1 = (float*)AllocateAndCopyToDevice(g->C1, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(C1, &(dg->C1), size, 0, cudaMemcpyHostToDevice));
    dg->C2 = (float*)AllocateAndCopyToDevice(g->C2, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(C2, &(dg->C2), size, 0, cudaMemcpyHostToDevice));

    printf("Done!\n");
    return;
}

void InitGridDeltas(grid *g, grid *dg)
{
    printf("Initializing mesh deltas...");

    g->dEx = (float *)cust_alloc(sizeof(float) * g->nx);
    g->dEy = (float *)cust_alloc(sizeof(float) * g->ny);
    g->dEz = (float *)cust_alloc(sizeof(float) * g->nz);
    g->dHx = (float *)cust_alloc(sizeof(float) * g->nx);
    g->dHy = (float *)cust_alloc(sizeof(float) * g->ny);
    g->dHz = (float *)cust_alloc(sizeof(float) * g->nz);

    //compute the deltax,y,z for each field - for nonuniform meshing (TODO: add mesh points in json file)
    //renormalized values as dt/dx,y,z to avoid divisions in cuda
    for (unsigned int i = 0; i < g->nx; i++)
    {
        if (i < (g->nx - 1))
            g->dHx[i] = 2 / (g->dx + g->dx); // should be g->dx[n] + g->dx[n+1], since we are 1/2 cell offset from E
        else
            g->dHx[i] = 1 / g->dx; //should be g->dx[n]

        g->dEx[i] = 1 / g->dx; //should be g->dx[n]
    }

    for (unsigned int i = 0; i < g->ny; i++)
    {
        if (i < (g->ny - 1))
            g->dHy[i] = 2 / (g->dy + g->dy);
        else
            g->dHy[i] = 1 / g->dy;

        g->dEy[i] = 1 / g->dy;
    }

    for (unsigned int i = 0; i < g->nz; i++)
    {
        if (i < (g->nz - 1))
            g->dHz[i] = 2 / (g->dz + g->dz);
        else
            g->dHz[i] = 1 / g->dz;

        g->dEz[i] = 1 / g->dz;
    }

    size_t size = sizeof(float)*g->nx;
    dg->dEx = (float*)AllocateAndCopyToDevice(g->dEx, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(DEX, &(dg->dEx), size, 0, cudaMemcpyHostToDevice));
    dg->dHx = (float*)AllocateAndCopyToDevice(g->dHx, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(DHX, &(dg->dHx), size, 0, cudaMemcpyHostToDevice));

    size = sizeof(float)*g->ny;
    dg->dEy = (float*)AllocateAndCopyToDevice(g->dEy, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(DEY, &(dg->dEy), size, 0, cudaMemcpyHostToDevice));
    dg->dHy = (float*)AllocateAndCopyToDevice(g->dHy, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(DHY, &(dg->dHy), size, 0, cudaMemcpyHostToDevice));

    size = sizeof(float)*g->nz;
    dg->dEz = (float*)AllocateAndCopyToDevice(g->dEz, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(DEZ, &(dg->dEz), size, 0, cudaMemcpyHostToDevice));
    dg->dHz = (float*)AllocateAndCopyToDevice(g->dHz, size);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(DHZ, &(dg->dHz), size, 0, cudaMemcpyHostToDevice));

    printf("Done!\n");
    return;
}


//copies all fields from device memory back to host
bool CudaRetrieveAll(grid *g, grid *dg)
{
    CudaRetrieveField(g->ex, dg->ex, sizeof(float)*g->domainSize);
    CudaRetrieveField(g->ey, dg->ey, sizeof(float)*g->domainSize);
    CudaRetrieveField(g->ez, dg->ez, sizeof(float)*g->domainSize);
    CudaRetrieveField(g->hx, dg->hx, sizeof(float)*g->domainSize);
    CudaRetrieveField(g->hy, dg->hy, sizeof(float)*g->domainSize);
    CudaRetrieveField(g->hz, dg->hz, sizeof(float)*g->domainSize);
    return true;
}


//copies a specific field from device memory to host - Async version
void CudaRetrieveField_Async(float *h_data, float *d_data, unsigned long size, cudaStream_t stream)
{
    CopyToHost_Async(d_data, h_data, size, stream);
}

//copies a specific field from device memory to host
void CudaRetrieveField(float *h_data, float *d_data, unsigned long size)
{
    CopyToHost(d_data, h_data, size);
}


//free memory on device for all the arrays
void CudaFreeFields(grid *g)
{
    cudaFree(g->ex);     cudaFree(g->ey);     cudaFree(g->ez);
    cudaFree(g->hx);	 cudaFree(g->hy);     cudaFree(g->hz);
    cudaFree(g->srcField);
    cudaFree(g->detEx);  cudaFree(g->detEy);  cudaFree(g->detEz);
    cudaFree(g->detHx);  cudaFree(g->detHy);  cudaFree(g->detHz);
    cudaFree(g);
    return;

}