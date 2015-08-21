#include "SimCuda3D\Cuda_macros.h"
#include "SimCuda3D\SimCudaFunctions.h"
#include "SimCuda3D\Cuda_grid.h"
#include "simCuda3D\Cuda_memory.h"
#include "cuda_runtime.h"

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




/*************** host side functions ***************************/

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
    dg->detEx = (float*)AllocateAndCopyToDevice(g->detEx, sizeof(float)*g->nt);
    //allocate and copy constants to constant memory (limit: 64KB)

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NX, &(g->nx), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NY, &(g->ny), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NZ, &(g->nz), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(DOMAINSIZE, &(g->domainSize), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    g->srclinpos = ((g->srcposX) + (g->srcposY)* (g->nx) + (g->srcposZ)* (g->nx)* (g->ny));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(SRCLINPOS, &(g->srclinpos), sizeof(int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(SRCFIELDCOMP, &(g->srcFieldComp), sizeof(int), 0, cudaMemcpyHostToDevice));
    return true;
}

//retrieves block and grid sizes from the card parameters
void CudaGetBlockSize(unsigned int &blocksize, unsigned int &gridsize, grid *g)
{
    int maxThreadsPerBlock;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));
    blocksize = maxThreadsPerBlock;

    //number of blocks to use
    if (g->domainSize < blocksize) //very small domain
    {
        gridsize = 1;
    }
    else
    {
        gridsize = g->domainSize / blocksize;
        if (g->domainSize % blocksize) gridsize++; //if not a multiple of BlockSize, we add 1 block
    }
}

//do all calculations for the current timestep
bool CudaCalculateStep(int blocksize, int gridsize, grid *g, unsigned long iteration)
{

    Cuda_updateE << < gridsize, blocksize >> >(g->ex, g->ey, g->ez, g->hx, g->hy, g->hz, g->Ca, g->Cb1, g->Cb2, g->mat, g->srcField, iteration);
    CUDA_CHECK(cudaPeekAtLastError());
    Cuda_updateH << < gridsize, blocksize >> >(g->ex, g->ey, g->ez, g->hx, g->hy, g->hz, g->Db1, g->Db2, g->mat);
    CUDA_CHECK(cudaPeekAtLastError());
    Cuda_CaptureFields << < 1, 1 >> >(g->ex, g->ey, g->ez, g->hx, g->hy, g->hz, g->detEx, iteration);
    CUDA_CHECK(cudaPeekAtLastError());
    return true;
}


//writes the time series data to file for debug purposes
bool CudaWriteTimeSeriesData(char* filename, float *det_data, unsigned long nt)
{
    FILE *f = fopen(filename, "w");
    if (f == NULL)
    {
        printf("CudaWriteTimeSeriesData: Cannot open file for writing->\n");
        exit(1);
    }

    //write header
    int min = 0; int max = 0;
    fprintf(f, "BCF2DPC\n");
    fprintf(f, "%d\n", nt);
    fprintf(f, "%d %d\n", min, max);

    //write data
    for (unsigned int t = 0; t < nt; t++)
    {
            fprintf(f, "%d %f\n", t, det_data[t]);
    }

    fclose(f);

    return true;
}

//fills the frame with values from the fields
void PrepareFrame(grid *g, unsigned char * buf)
{
    int pos = 0;
    int i;
    double maxcolors = 1;
    double mincolors = 0.0;
    double normfact = 255.0 / (maxcolors - mincolors);
    for (int y = 0; y < g->ny; y++)
        for (int x = 0; x <g->nx; x++)
        {
            pos = 3 * (y * g->nx + x);
            i = ((x)+(y)* (g->nx) + (g->nz / 2)* (g->nx)* (g->ny));
            double val = (g->ex[i] - mincolors) * normfact;
            buf[pos] = (unsigned char)red(val); //R
            buf[pos + 1] = (unsigned char)green(val); //G
            buf[pos + 2] = (unsigned char)blue(val); //B
        }
}


double interpolate(double val, double y0, double x0, double y1, double x1) {
    return (val - x0)*(y1 - y0) / (x1 - x0) + y0;
}

double base(double val) {
    if (val <= -0.75) return 0;
    else if (val <= -0.25) return interpolate(val, 0.0, -0.75, 1.0, -0.25);
    else if (val <= 0.25) return 1.0;
    else if (val <= 0.75) return interpolate(val, 1.0, 0.25, 0.0, 0.75);
    else return 0.0;
}

double red(double gray) {
    return base(gray - 0.5) * 255;
}
double green(double gray) {
    return base(gray) * 255;
}
double blue(double gray) {
    return base(gray + 0.5) * 255;
}



/*************** device side functions ************************/

//H field update equations helper
__device__ inline void Cuda_updateHComponent(int component, float *h, int i, int j, int k, int pos, float *ex, float *ey, float *ez, float *Db1, float *Db2, unsigned int *mat)
{
    float e1a, e1b, e2a, e2b;

    //apply the boundaries
    if (mat[pos] == MAT_PEC)
    {
        h[pos] = 0.0;
        return;
    }

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

//E field update equations helper
__device__ inline void Cuda_updateEComponent(int component, float *e, int i, int j, int k, int pos, float *hx, float *hy, float *hz, float *Ca, float *Cb1, float *Cb2, unsigned int *mat, float srcfield)
{
    float h1a, h1b, h2a, h2b;

    //apply the boundaries
    if (mat[pos] == MAT_PEC)
    {
        e[pos] = 0.0;
        return;
    }

    switch (component)
    {
    case 0:
        h1a = hz[IDX(i, j + 1, k)];
        h1b = hz[pos];
        h2a = hy[IDX(i, j, k + 1)];
        h2b = hy[pos];
        break;
    case 1:
        h1a = hx[IDX(i, j, k + 1)];
        h1b = hx[pos];
        h2a = hz[IDX(i + 1, j, k)];
        h2b = hz[pos];
        break;
    case 2:
        h1a = hy[IDX(i + 1, j, k)];
        h1b = hy[pos];
        h2a = hx[IDX(i, j + 1, k)];
        h2b = hx[pos];
        break;
    }
    e[pos] = Ca[pos] * e[pos] + Cb1[pos] * (h1a - h1b) - Cb2[pos] * (h2a - h2b);

    //after E update, we add the E source
    if (pos == SRCLINPOS)
    {
        e[pos] += srcfield;
    }
}

//update equations for H fields
__global__ void Cuda_updateH(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *Db1, float *Db2, unsigned int *mat)
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

        Cuda_updateHComponent(XX, hx, i, j, k, pos, ex, ey, ez, Db1, Db2, mat);
        Cuda_updateHComponent(YY, hy, i, j, k, pos, ex, ey, ez, Db1, Db2, mat);
        Cuda_updateHComponent(ZZ, hz, i, j, k, pos, ex, ey, ez, Db1, Db2, mat);
    }
    return;
}

//update equations for E fields
__global__ void Cuda_updateE(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *Ca, float *Cb1, float *Cb2, unsigned int *mat, float *srcField, unsigned long iteration)
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

        Cuda_updateEComponent(XX, ex, i, j, k, pos, hx, hy, hz, Ca, Cb1, Cb2, mat, (SRCFIELDCOMP == XX? srcField[iteration] : 0.0f));
        Cuda_updateEComponent(YY, ey, i, j, k, pos, hx, hy, hz, Ca, Cb1, Cb2, mat, (SRCFIELDCOMP == YY? srcField[iteration] : 0.0f));
        Cuda_updateEComponent(ZZ, ez, i, j, k, pos, hx, hy, hz, Ca, Cb1, Cb2, mat, (SRCFIELDCOMP == ZZ? srcField[iteration] : 0.0f));

        
    }
    return;
}


__global__ void Cuda_CaptureFields(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *detEx, unsigned long iteration)
{
  
    detEx[iteration] = ex[SRCLINPOS];
  
}



