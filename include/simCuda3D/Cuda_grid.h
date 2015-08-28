//Cuda_grid.h: FDTD grid structure definitions
// Aurelien Duval 2015

#pragma once
#ifndef _CUDA_GRID_H_
#define _CUDA_GRID_H_

//#include "cuda_runtime.h"

//to code and retrieve field components in detectors
//for gcc, use "-fms-extensions" or name the struct to avoid errors
#pragma warning(disable : 4201) // compiler doesn't like unamed structs

union FieldComps
{
    struct {
        unsigned __int8 Ex : 1;
        unsigned __int8 Ey : 1;
        unsigned __int8 Ez : 1;
        unsigned __int8 Hx : 1;
        unsigned __int8 Hy : 1;
        unsigned __int8 Hz : 1;
    };
    unsigned __int8 comps;
};

//material ids for mat array. boundaries and sources are also coded here since we have a uniform array at the end
#define MAT_DIEL        1
#define MAT_DIEL_ANISO  2
#define MAT_LD          3

#define MAT_PML         10
#define MAT_PEC         11
#define MAT_PBC         12

#define MAT_SRC         99

#define MAX_MAT         9

#define XX              0
#define YY              1
#define ZZ              2

#define TILEXX          16
#define TILEYY          16


#define tx threadIdx.x
#define ty threadIdx.y

#define bx blockIdx.x
#define by blockIdx.y

#define EPSILON_0 8.854187817E-12
#define MU_0  1.256637061435917e-06
#define MU_0INV 7.957747154594768e+05

#define MAX_SIM_SIZE 2048

typedef struct _grid{
    //fields and material arrays
    float * ex; //field components arrays
    float * ey;
    float * ez;
    float * hx;
    float * hy;
    float * hz;
    unsigned int * mat; // material IDs array 
    float * epsilon; //epsilon values corresponding to mat ids
    unsigned int Nmats; //number of different materials used

    //coefficients arrays
    float * C1;  // (1 - sigma*dt/2*epsilon) / (1 + sigma*dt/2*epsilon) array
    float * C2; // (dt/epsilon*delta1) / (1 + sigma*dt/2*epsilon) array
   // float * Cb2; // (dt/epsilon*delta2) / (1 + sigma*dt/2*epsilon) array
    //float * Db1; // (dt/mu*delta1)
    //float * Db2; // (dt/mu*delta2)

    //source arrays (1 source only for now)
    float * srcField; //auxiliary array for the source amplitude vs time (precomputed)
    unsigned int srclinpos; //linear position of the source
    unsigned int srcposX; //indexed position of the source
    unsigned int srcposY;
    unsigned int srcposZ;
    unsigned int srcFieldComp; //Source field component (0: Ex, 1: Ey, 2: Ez)
    double amplitude; //in V/m
    double omega; // in rad/s
    double rTime; // ramp time sec
    double initPhase; //phase (radian)

    //detector arrays
    float *detEx, *detEy, *detEz;
    float *detHx, *detHy, *detHz;
    unsigned int detX, detY, detZ;
    unsigned __int8 detComps; //bitfield giving active detectors

    //grid properties
    unsigned int nx, ny, nz, domainSize; //grid size
    unsigned int layoutx, layouty, layoutz; //nb of cells in layout
    float *dEx, *dEy, *dEz; //normalized deltas (dt/dx, dt/dy, dt/dz). Suitable for non-uniform mesh. Avoids divisions in kernels
    float *dHx, *dHy, *dHz; //normalized deltas (dt/(mu*dx), dt/(mu*dy), dt/(mu*dz)).


    int offset; //used to correspond to material indices in OptiFDTD
    unsigned long nt; // number of time steps
    float dt; //time step size (sec)
    float dx, dy, dz; // mesh cell size (m)
    unsigned long currentIteration; //currently running iteration 

    //cuda launch properties
  //  dim3 BlockSize;
  //  dim3 GridSize;
    
} grid;



#endif /*_CUDA_GRID_H_*/