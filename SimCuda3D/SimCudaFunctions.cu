#include "SimCuda3D\Cuda_macros.h"
#include "SimCuda3D\Cuda_protos.h" 
#include "SimCuda3D\SimCudaFunctions.h"
#include "cuda_runtime.h"
#include "SimCuda3D\Cuda_grid.h"
#include <vector>

cudaStream_t Stream;
dim3 GridSize;
dim3 BlockSize;

//allocates the grid struct in unified memory to avoid deep copy issues later
grid *CudaInitGrid()
{
    //run all in a single stream
    cudaStreamCreate(&Stream);

    grid *g = NULL;
    CUDA_ALLOC_MANAGED_1D(g, 1, grid);

    //attaches the grid to current stream
    cudaStreamAttachMemAsync(Stream, g);
    HANDLE_ERROR(cudaStreamSynchronize(Stream));    
    return g;
}

//initializes the host and device memory chunks for simulation
void CudaInitFields(grid *g)
{
    //query the card to use the max number of threads per block possible
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    BlockSize.x = maxThreadsPerBlock; 

    //number of blocks to use
    if (g->domainSize < BlockSize.x) //very small domain
    {
        GridSize.x = 1;
    }
    else
    {
        GridSize.x = g->domainSize / BlockSize.x;
        if (g->domainSize % BlockSize.x) GridSize.x++; //if not a multiple of BlockSize, we add 1 block
    }

    
    CudaAllocateFields(g); // allocate on the device
    
    //initialize all to zero
    Cuda_initFieldArrays <<< GridSize, BlockSize, 0, Stream >>>( g);
    HANDLE_ERROR(cudaStreamSynchronize(Stream));
}


bool CudaInitDetectorComponent(unsigned __int64 nbTimeSteps, float *fieldcomp)
{
    size_t free_mem = 0;
    size_t total_mem = 0;
    unsigned __int64 tsteps = 0;
    bool success = true;

    //do we have enough space left on device?
    HANDLE_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    if (free_mem < sizeof(float) * nbTimeSteps)
    {
        printf("CudaInitDetectors: Not enough free space on device for detector! Reducing the size...\n");
        tsteps = 1000;//(free_mem / sizeof(float) - 100*sizeof(float)); //we allocate what we can
        success = false;
    }
    CUDA_ALLOC_MANAGED_1D(fieldcomp, tsteps, float); HANDLE_ERROR(cudaDeviceSynchronize());
    for (int i = 0; i < tsteps; i++) fieldcomp[i] = 0.0;
    return success;
}

//initialization of the point detector time series data
//nbTimeSteps: total number of timesteps
//detComps: field components selected (see FieldComps union)
void CudaInitDetectors(grid *g)
{
    union FieldComps fc;
    fc.comps = g->detComps;
    
    //time series detection
    if (fc.Ex)
    {   
        CudaInitDetectorComponent(g->nt, g->detEx);
    }
    if (fc.Ey)
    {
        CudaInitDetectorComponent(g->nt, g->detEy);
    }
    if (fc.Ez)
    {
        CudaInitDetectorComponent(g->nt, g->detEz);
    }
    if (fc.Hx)
    {
        CudaInitDetectorComponent(g->nt, g->detHx);
    }
    if (fc.Hy)
    {
        CudaInitDetectorComponent(g->nt, g->detHy);
    }
    if (fc.Hz)
    {
        CudaInitDetectorComponent(g->nt, g->detHz);
    }

}


//Initialize the E source auxiliary array
void CudaInitializeSourceE(grid *g)
{
    CUDA_ALLOC_MANAGED_1D(g->srcField, g->nt, float);  cudaStreamAttachMemAsync(Stream, g->srcField);
     
    HANDLE_ERROR(cudaStreamSynchronize(Stream));
    Cuda_InitializeSrc << < GridSize, BlockSize,  0, Stream >> > (g);
    HANDLE_ERROR(cudaStreamSynchronize(Stream));
}


//allocate memory in unified memory (CUDA >6.0) for all the arrays
void CudaAllocateFields(grid *g)
{
    CUDA_ALLOC_MANAGED_3D(g->ex, g->nx, g->ny, g->nz, float); cudaStreamAttachMemAsync(Stream, g->ex);
    CUDA_ALLOC_MANAGED_3D(g->ey, g->nx, g->ny, g->nz, float); cudaStreamAttachMemAsync(Stream, g->ey);
    CUDA_ALLOC_MANAGED_3D(g->ez, g->nx, g->ny, g->nz, float); cudaStreamAttachMemAsync(Stream, g->ez);
    CUDA_ALLOC_MANAGED_3D(g->Ca, g->nx, g->ny, g->nz, float); cudaStreamAttachMemAsync(Stream, g->Ca);
    CUDA_ALLOC_MANAGED_3D(g->Cb1, g->nx, g->ny, g->nz, float); cudaStreamAttachMemAsync(Stream, g->Cb1);
    CUDA_ALLOC_MANAGED_3D(g->Cb2, g->nx, g->ny, g->nz, float); cudaStreamAttachMemAsync(Stream, g->Cb2);
    CUDA_ALLOC_MANAGED_3D(g->hx, g->nx, g->ny, g->nz, float); cudaStreamAttachMemAsync(Stream, g->hx);
    CUDA_ALLOC_MANAGED_3D(g->hy, g->nx, g->ny, g->nz, float); cudaStreamAttachMemAsync(Stream, g->hy);
    CUDA_ALLOC_MANAGED_3D(g->hz, g->nx, g->ny, g->nz, float); cudaStreamAttachMemAsync(Stream, g->hz);
    CUDA_ALLOC_MANAGED_3D(g->Db1, g->nx, g->ny, g->nz, float); cudaStreamAttachMemAsync(Stream, g->Db1);
    CUDA_ALLOC_MANAGED_3D(g->Db2, g->nx, g->ny, g->nz, float); cudaStreamAttachMemAsync(Stream, g->Db2);
    HANDLE_ERROR(cudaStreamSynchronize(Stream));

    return;
}

//free memory on device for all the arrays
void CudaFreeFields(grid *g)
{
    cudaStreamSynchronize(Stream);
    cudaStreamDestroy(Stream);

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
    Cuda_CalculateStep << < GridSize, BlockSize, 0, Stream >> >(g);
    HANDLE_ERROR(cudaStreamSynchronize(Stream));
    return true;
}

//Collect time series data for the current point detector
//fComps: field components selected (see FieldComps union)
//timestep: current timestep
//detX, detY, detZ: position of the detector
bool CudaCollectTimeSeriesData( __int64 timestep, int detX, int detY, int detZ, grid *g)
{
    HANDLE_ERROR(cudaDeviceSynchronize());

    union FieldComps fc; 
    fc.comps= g->detComps;
    
    //only 1 thread since we only want the value at the detector point
    if (fc.Ex)
        Cuda_CollectTimeSeriesData << <1, 1 >> >(g->detEx, g->ex, detX, detY, detZ, timestep,g);
    if (fc.Ey)                                                                        
        Cuda_CollectTimeSeriesData << <1, 1 >> >(g->detEy, g->ey, detX, detY, detZ, timestep,g);
    if (fc.Ez)                                                                        
        Cuda_CollectTimeSeriesData << <1, 1 >> >(g->detEz, g->ez, detX, detY, detZ, timestep,g);
    if (fc.Hx)                                                                        
        Cuda_CollectTimeSeriesData << <1, 1 >> >(g->detHx, g->hx, detX, detY, detZ, timestep,g);
    if (fc.Hy)                                                                        
        Cuda_CollectTimeSeriesData << <1, 1 >> >(g->detHy, g->hy, detX, detY, detZ, timestep,g);
    if (fc.Hz)                                                                        
        Cuda_CollectTimeSeriesData << <1, 1 >> >(g->detHz, g->hz, detX, detY, detZ, timestep,g);

    return true;
}

//writes the time series data to file for debug purposes
bool CudaWriteTimeSeriesData(char* filename, grid *g)
{
    union FieldComps fc;
    fc.comps = g->detComps;
    HANDLE_ERROR(cudaDeviceSynchronize());
    FILE *f = fopen(filename, "w");
    if (f == NULL)
    {
        printf("CudaWriteTimeSeriesData: Cannot open file for writing->\n");
        exit(1);
    }

    //write header
    int min = 0; int max = 0;
    fprintf(f, "BCF2DPC\n");
    fprintf(f, "%d\n", g->nt);
    fprintf(f, "%d %d\n", min, max);

    //write data
    for (unsigned int t = 0; t < g->nt; t++)
    {
      //  if (fc.Ex)
     //       fprintf(f, "%d %f\n", t, g->detEx[t]);
       /* if (fc.Ey)
            fprintf(f, "%d %f", t, detEy[t]);
        if (fc.Ez)
            fprintf(f, "%d %f", t, detEz[t]);
        if (fc.Hx)
            fprintf(f, "%d %f", t, detHx[t]);
        if (fc.Hy)
            fprintf(f, "%d %f", t, detHy[t]);
        if (fc.Hz)
            fprintf(f, "%d %f", t, detHz[t]);*/
    }

    fclose(f);

    return true;
}

/*
//Multiplex time series data for use in DS_Observer
std::vector< float > CudaMultiplexTimeSeriesData(unsigned __int8 fcomps, int nbTimeSteps)
{
    union FieldComps fc;
    fc.comps = fcomps;
    std::vector< float > timeseries;
    
    //make sure we have access to the arrays
    HANDLE_ERROR(cudaDeviceSynchronize());
   
    //for some weird reason (probably memory usage) all fields data points are 
    //interleaved inside a single vector....
    for (int t = 0; t < nbTimeSteps; t++)
    {
        if (fc.Ex)
            timeseries.push_back(detEx[t]);
        if (fc.Ey)
            timeseries.push_back(detEy[t]);
        if (fc.Ez)
            timeseries.push_back(detEz[t]);
        if (fc.Hx)
            timeseries.push_back(detHx[t]);
        if (fc.Hy)
            timeseries.push_back(detHy[t]);
        if (fc.Hz)
            timeseries.push_back(detHz[t]);
    }

    return timeseries;
}
*/