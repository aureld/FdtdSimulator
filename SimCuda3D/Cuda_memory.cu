#include "simCuda3D/Cuda_memory.h"
#include "simCuda3D/Cuda_macros.h"

//allocates a chunck of global memory on device and copy the data there
void* AllocateAndCopyToDevice(void *h_data, unsigned int memsize)
{
    // allocate device global memory
    void *d_data;
    CUDA_SAFE_CALL(cudaMalloc((void**)& d_data, memsize));
    // copy to device
    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, memsize, cudaMemcpyHostToDevice));
    return d_data;
}

//copies a chunk of global memory from device to host memory
void CopyToHost(void *d_data, void *h_data, unsigned int memsize)
{
    CUDA_SAFE_CALL(cudaMemcpy(h_data, d_data, memsize, cudaMemcpyDeviceToHost));
}


//allocates the device fields in global memory
bool CudaInitFields(grid *g, grid *dg)
{
    if (g->ex == NULL || g->ey == NULL || g->ez == NULL || g->hx == NULL || g->hy == NULL || g->hz == NULL)
    {
        perror("[CudaInitGrid]: host fields must be initialized");
        return false;
    }

    dg->ex = (float*)AllocateAndCopyToDevice(g->ex, sizeof(float)*g->domainSize);
    dg->ey = (float*)AllocateAndCopyToDevice(g->ey, sizeof(float)*g->domainSize);
    dg->ez = (float*)AllocateAndCopyToDevice(g->ez, sizeof(float)*g->domainSize);
    dg->hx = (float*)AllocateAndCopyToDevice(g->hx, sizeof(float)*g->domainSize);
    dg->hy = (float*)AllocateAndCopyToDevice(g->hy, sizeof(float)*g->domainSize);
    dg->hz = (float*)AllocateAndCopyToDevice(g->hz, sizeof(float)*g->domainSize);

    if (g->mat == NULL)
    {
        perror("[CudaInitGrid]: host material grid must be initialized");
        return false;
    }
    dg->mat = (unsigned int*)AllocateAndCopyToDevice(g->mat, sizeof(unsigned int)*g->domainSize);


    return true;

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


//copies a specific field from device memory to host
inline void CudaRetrieveField(float *h_data, float *d_data, unsigned long size)
{
    CopyToHost(d_data, h_data, size);
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