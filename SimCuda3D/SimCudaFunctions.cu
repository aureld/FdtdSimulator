#include "SimCuda3D\Cuda_macros.h"
#include "SimCuda3D\SimCudaFunctions.h"
#include "SimCuda3D\Cuda_grid.h"
#include "simCuda3D\Cuda_memory.h"
//#include "simCuda3D/Cuda_constantMemoryData.h"
#include "common_defs.h"
#include "cuda_runtime.h"

extern __constant__ unsigned int NX;
extern __constant__ unsigned int NY;
extern __constant__ unsigned int NZ;
extern __constant__ unsigned int DOMAINSIZE;
extern __constant__ int    SRCLINPOS;
extern __constant__ int    SRCFIELDCOMP;
extern __constant__ float DT;
extern __constant__ float *EX;
extern __constant__ float *EY;
extern __constant__ float *EZ;
extern __constant__ float *HX;
extern __constant__ float *HY;
extern __constant__ float *HZ;
extern __constant__ float *DETEX;
extern __constant__ unsigned int *MAT;
extern __constant__ float EPSILON[MAX_SIM_SIZE];
extern __constant__ float DEX[MAX_SIM_SIZE];
extern __constant__ float DEY[MAX_SIM_SIZE];
extern __constant__ float DEZ[MAX_SIM_SIZE];
extern __constant__ float DHX[MAX_SIM_SIZE];
extern __constant__ float DHY[MAX_SIM_SIZE];
extern __constant__ float DHZ[MAX_SIM_SIZE];
extern __constant__ float C1[MAX_NB_MAT];
extern __constant__ float C2[MAX_NB_MAT];

/*************** host side functions ***************************/


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

   // Cuda_updateE << < gridsize, blocksize >> >(iteration);
   // CUDA_CHECK(cudaPeekAtLastError());
    Cuda_updateH << < gridsize, blocksize >> >(iteration);
    CUDA_CHECK(cudaPeekAtLastError());
   // Cuda_CaptureFields << < 1, 1 >> >(iteration);
   // CUDA_CHECK(cudaPeekAtLastError());
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

__device__ void GetEShared(float E[TILEXX + 1][TILEXX + 1], int i, int j, int k, float *field)
{
    E[tx][ty] = field[IDX(i,j,k)]; //load the field tile into shared memory

    if (tx == TILEXX - 1) {
        if (bx == (NX / TILEXX) - 1)
            E[tx + 1][ty] = 0;
        else
            E[tx + 1][ty] = field[IDX(i + 1, j, k)];

    }
    if (ty == TILEYY - 1) {
        if (by == (NY / TILEYY) - 1)
            E[tx][ty + 1] = 0;
        else
            E[tx][ty + 1] = field[IDX(i, j + 1, k)];
    }
    return;
}


__device__ void zeroShared(float A[TILEXX + 1][TILEYY + 1])
{
    A[tx + 1][ty + 1] = 0;
    if (tx == 0)
        A[tx][ty + 1] = 0;
    if (ty == 0)
        A[tx + 1][ty] = 0;
    return;
}

__device__ void SwapShared_H(float A[TILEXX + 1][TILEYY + 1], float B[TILEXX + 1][TILEYY + 1]) 
{
    B[tx][ty] = A[tx][ty];
    if (tx == TILEXX - 1)
        B[tx + 1][ty] = A[tx + 1][ty];
    if (ty == TILEYY - 1)
        B[tx][ty + 1] = A[tx][ty + 1];
    return;
}



//update equations for H fields
__global__ void Cuda_updateH(unsigned long iteration)
{
    int i, j, k;

    __shared__ float Ex_a[TILEXX + 1][TILEYY + 1]; 
    __shared__ float Ey_a[TILEXX + 1][TILEYY + 1];
    __shared__ float Ez_a[TILEXX + 1][TILEYY + 1]; 
    __shared__ float Ex_b[TILEXX + 1][TILEYY + 1]; 
    __shared__ float Ey_b[TILEXX + 1][TILEYY + 1];

    i = tx + bx*TILEXX;
    j = ty + by*TILEYY;

    //retrieve the field from global mem and load into the current tile
    //layer 0 (k = 0)
    GetEShared(Ex_a, i, j, 0, EX);
    GetEShared(Ey_a, i, j, 0, EY);

    //loop on z for each thread
    for (k = 0; k < NZ; k++)
    {
        GetEShared(Ez_a, i, j, k, EZ); //Ez at current layer

        if (k == (NZ - 1)) //if we are in last layer
        {
            zeroShared(Ex_b); //put the E fields to 0 since k+1 doesn't exist
            zeroShared(Ey_b);
        }
        else
        {
            GetEShared(Ex_b, i, j, k+1, EX);
            GetEShared(Ey_b, i, j, k+1, EY);
        }

        __syncthreads(); //barrier before the main calculation

        unsigned int mat = MAT[IDX(i, j, k)];

        Cuda_updateHComponent(XX, HX, i ,j ,k, Ex_a, Ex_b, Ey_a, Ey_b, Ez_a, mat);
        Cuda_updateHComponent(YY, HY, i, j, k, Ex_a, Ex_b, Ey_a, Ey_b, Ez_a, mat);
        Cuda_updateHComponent(ZZ, HZ, i, j, k, Ex_a, Ex_b, Ey_a, Ey_b, Ez_a, mat);

        __syncthreads();

        SwapShared_H(Ex_b, Ex_a);
        SwapShared_H(Ey_b, Ey_a);

    }
    return;
}

//H field update equations helper
__device__ inline void Cuda_updateHComponent(   int component, float *h, int i0, int j0, int k0,
                                                float Ex_a[TILEXX + 1][TILEYY + 1], float Ex_b[TILEXX + 1][TILEYY + 1],
                                                float Ey_a[TILEXX + 1][TILEYY + 1], float Ey_b[TILEXX + 1][TILEYY + 1],
                                                float Ez_a[TILEXX + 1][TILEYY + 1], unsigned int mat)
{
    float e1a, e1b, e2a, e2b, delta1, delta2;
    
    float H = h[IDX(i0, j0, k0)]; // retrieve the current h value from global mem and load into register

    float HC = DT*MU_0INV;
    int i = tx;
    int j = ty;

    switch (component)
    {
    case XX: 
        e1a = Ez_a[i][j+1];
        e1b = Ez_a[i][j];
        e2a = Ey_a[i][j];
        e2b = Ey_a[i][j];
        delta1 = DHY[j0];
        delta2 = DHZ[k0];
        break;
    case YY: 
        e1a = Ex_b[i][j];
        e1b = Ex_a[i][j];
        e2a = Ez_a[i+1][j];
        e2b = Ez_a[i][j];
        delta1 = DHZ[k0];
        delta2 = DHX[i0];
        break;
    case ZZ: 
        e1a = Ey_a[i+1][j];
        e1b = Ey_a[i][j];
        e2a = Ex_a[i][j+1];
        e2b = Ex_a[i][j];
        delta1 = DHX[i0];
        delta2 = DHY[j0];
        break;
    }

    //H update
    H = H - HC * (delta1 * (e1a - e1b) - delta2* (e2a - e2b));

    //apply the boundaries
    if (mat == MAT_PEC)
        H = 0.0;

    h[IDX(i0, j0, k0)] = H; //transfer results back to global mem
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



