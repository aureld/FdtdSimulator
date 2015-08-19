// runFDTD.cu : FDTD core engine in C, CUDA variant
// see http://www.eecs.wsu.edu/~schneidj
// Aurelien Duval 2015

#include <stdio.h>
#include <iostream>
#include "SimCuda3D/Cuda_grid.h"
#include "SimCuda3D/Cuda_macros.h"
#include "SimCuda3D/SimCudaFunctions.h"
#include "JsonParser/JsonParser.h"
#include "JsonParser/JsonDocument.h"
#include "JsonParser/FileIO.h"
#include "common_defs.h"
#include <math.h>

using namespace std;


void main() 
{
    grid *g, *dg; //g is the host grid, dg is the device grid
    g = new grid();
    dg = new grid();
    JsonDocument *doc = new JsonDocument();
    FileIO *fio = new FileIO();
    JsonParser *parser = new JsonParser(doc, fio);
    parser->ParseJsonFile("ValidTestJson.txt", g); //retrieve all parameters and material data from json file
    
    //create the field arrays
    g->ex = (float *)cust_alloc(sizeof(float)*g->domainSize);
    g->ey = (float *)cust_alloc(sizeof(float)*g->domainSize);
    g->ez = (float *)cust_alloc(sizeof(float)*g->domainSize);
    g->hx = (float *)cust_alloc(sizeof(float)*g->domainSize);
    g->hy = (float *)cust_alloc(sizeof(float)*g->domainSize);
    g->hz = (float *)cust_alloc(sizeof(float)*g->domainSize);

    //source field data is amplitude vs time (precomputed amplitude)
    g->srcField = (float *)cust_alloc(sizeof(float)*g->nt);

    //initialize the source amplitude data
    for (int index = 0; index < g->nt; index++)
    {
        double d_efftime = index * g->dt;
        double envelope = 1.0 - exp(-(d_efftime / g->rTime)); //CW for now
        g->srcField[index] = g->amplitude * envelope * sin(g->omega * d_efftime + g->initPhase);
    }


    if (CudaInitGrid(g, dg) == false)
    {
        perror("Cuda grid initialization error.");
        exit(-1);
    }
    if (CudaInitFields(g, dg) == false)
    {
        perror("Cuda fields initialization error.");
        exit(-1);
    }


    //main FDTD loop
    printf("Starting simulation...\n");
   // for (g->currentIteration = 0; g->currentIteration < g->nt; g->currentIteration++)
    {
        printf("step %d / %d\n",g->currentIteration, g->nt);
        CudaCalculateStep(dg);
    }
    printf("\n");
    CudaFreeFields(dg);

    exit(0);
}