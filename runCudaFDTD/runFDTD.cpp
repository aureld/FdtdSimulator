// runFDTD.cu : FDTD core engine in C, CUDA variant
// Aurelien Duval 2015

#include "SimCuda3D/Cuda_grid.h"
#include "SimCuda3D/Cuda_macros.h"
#include "SimCuda3D/SimCudaFunctions.h"
#include "SimCuda3D/Cuda_memory.h"
#include "JsonParser/JsonParser.h"
#include "JsonParser/JsonDocument.h"
#include "JsonParser/FileIO.h"
#include "Movielib/MovieLib.h"
#include "common_defs.h"

using namespace std;


void main() 
{
    grid *g, *dg; //g is the host grid, dg is the device grid
    Movie *movie;
    g = new grid();
    dg = new grid();
    movie = new Movie();
    JsonDocument *doc = new JsonDocument();
    FileIO *fio = new FileIO();
    JsonParser *parser = new JsonParser(doc, fio);
    parser->ParseJsonFile("testSim.txt", g); //retrieve all parameters and material data from json file
    //create the field arrays
    g->ex = (float *)cust_alloc(sizeof(float)*g->domainSize);
    g->ey = (float *)cust_alloc(sizeof(float)*g->domainSize);
    g->ez = (float *)cust_alloc(sizeof(float)*g->domainSize);
    g->hx = (float *)cust_alloc(sizeof(float)*g->domainSize);
    g->hy = (float *)cust_alloc(sizeof(float)*g->domainSize);
    g->hz = (float *)cust_alloc(sizeof(float)*g->domainSize);

    //source field data is amplitude vs time (precomputed amplitude)
    g->srcField = (float *)cust_alloc(sizeof(float)*g->nt);

    //time series point detector initialization
    g->detEx = (float *)cust_alloc(sizeof(float)*g->nt);


    //initialize the source amplitude data
    for (unsigned long index = 0; index < g->nt; index++)
    {
        double d_efftime = index * g->dt;
        double envelope = 1.0 - exp(-(d_efftime / g->rTime)); //CW for now
        g->srcField[index] = g->amplitude * envelope * sin(g->omega * d_efftime + g->initPhase);
    }

    CudaWriteTimeSeriesData("srcEx.f2d", g->srcField, g->nt);


    //create the movie
    unsigned char *buf = new unsigned char[g->nx * g->ny * 3];
    movie->Initialize(g->nx, g->ny);
    movie->SetFileName("movie.avi");
    movie->Start();


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


    //calculate the grid and block size for kernel launches
    //query the card to use the max number of threads per block possible
    unsigned int blocksize = 0, gridsize = 0;
    CudaGetBlockSize(blocksize, gridsize, g);



    //main FDTD loop
    printf("Starting simulation...\n");
    for (g->currentIteration = 0; g->currentIteration < g->nt; g->currentIteration++)
    {
        CudaCalculateStep(blocksize, gridsize, dg, g->currentIteration);
        CudaRetrieveField(g->ex, dg->ex, sizeof(float)*g->domainSize);

        PrepareFrame(g, buf);
        movie->SetData(buf);
        movie->Write();

        if (g->currentIteration % 1000 == 0)
            printf("step %d / %d\n", g->currentIteration, g->nt-1);
    }

    movie->End();
    //retrieve all field data (final state)
   // CudaRetrieveAll(g, dg);
    CudaRetrieveField(g->detEx, dg->detEx, sizeof(float)*g->nt);
    CudaWriteTimeSeriesData("timeEx.f2d", g->detEx, g->nt);

    printf("\n");
    CudaFreeFields(dg);

    exit(0);
}