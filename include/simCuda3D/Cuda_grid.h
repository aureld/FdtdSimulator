//Cuda_grid.h: FDTD grid structure definitions
// Aurelien Duval 2015

#pragma once
#ifndef _CUDA_GRID_H_
#define _CUDA_GRID_H_

//#include "cuda_runtime.h"

typedef struct _grid{
    //fields and material arrays
    float * ex; //field components arrays
    float * ey;
    float * ez;
    float * hx;
    float * hy;
    float * hz;
    int   * mat; // material IDs array

    //coefficients arrays
    float * Ca;  // (1 - sigma*dt/2*epsilon) / (1 + sigma*dt/2*epsilon) array
    float * Cb1; // (dt/epsilon*delta1) / (1 + sigma*dt/2*epsilon) array
    float * Cb2; // (dt/epsilon*delta2) / (1 + sigma*dt/2*epsilon) array
    float * Db1; // (dt/mu*delta1)
    float * Db2; // (dt/mu*delta2)

    //source arrays (1 source only for now)
    float * srcField; //auxiliary array for the source amplitude vs time (precomputed)
    int srclinpos; //linear position of the source
    int srcposX; //indexed position of the source
    int srcposY;
    int srcposZ;
    int srcFieldComp; //Source field component (0: Ex, 1: Ey, 2: Ez)
    double amplitude; //in V/m
    double omega; // in rad/s
    double rTime; // ramp time sec
    double initPhase; //phase (radian)

    //detector arrays
    float *detEx, *detEy, *detEz;
    float *detHx, *detHy, *detHz;
    int detX, detY, detZ;
    unsigned __int8 detComps; //bitfield giving active detectors

    //grid properties
    unsigned int nx, ny, nz, domainSize; //grid size
    int layoutx, layouty, layoutz; //nb of cells in layout
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