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
#include <time.h>

using namespace std;


void main() 
{
    grid *g, *dg; //g is the host grid, dg is the device grid
    bool MOVIE = false;
   
    g = new grid();
    dg = new grid();
    Movie *movie;

    clock_t tstart, tcurrent, tprevious;
    int display_freq = 100;

    //create 1 stream for computations, and 1 for data transfer 
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&transfer_stream));


    //parse input data file
    JsonDocument *doc = new JsonDocument();
    FileIO *fio = new FileIO();
    JsonParser *parser = new JsonParser(doc, fio);
    parser->ParseJsonFile("testSim.txt", g); //retrieve all parameters and material data from json file
    
    //DEBUG
   // g->nt = 1000;

    if (CudaInitGrid(g, dg) == false)
    {
        perror("Cuda grid initialization error.");
        exit(-1);
    }
    
    //DEBUG - SHOW source field
    //CudaWriteTimeSeriesData("srcEx.f2d", g->srcField, g->nt);

    unsigned char *buf = new unsigned char[g->nx * g->ny * 3]; //for Movie, RGB colors so 3 values per point
    //create the movie
    if (MOVIE)
    {
        movie = new Movie();
        movie->Initialize(g->nx, g->ny);
        movie->SetFileName("movie.avi");
        movie->Start();
    }

    //calculate the grid and block size for kernel launches
    //query the card to use the max number of threads per block possible
    dim3 blocksize = 0, gridsize = 0;
    CudaGetBlockSize(blocksize, gridsize, g);

    //main FDTD loop
    printf("Starting simulation...\n");
    printf("Grid parameters: Block size = (%d, %d), grid size = (%d, %d)\n", blocksize.x, blocksize.y, gridsize.x, gridsize.y);
    printf("total grid size: %d\n", g->domainSize);

    //start the timer
    double speed, elapsed_time;

    tstart = clock();
    tcurrent = tstart;
    for (g->currentIteration = 0; g->currentIteration < g->nt; g->currentIteration++)
    {
        if (MOVIE)
        {
            CudaRetrieveField_Async(g->ex, dg->ex, sizeof(float)*g->domainSize, transfer_stream); //initiate copy while the kernel is running
        }

        CudaCalculateStep(blocksize, gridsize, dg, g->currentIteration, compute_stream); //hands off field updates to device
        

        if (MOVIE)
        {
            PrepareFrame(g, buf);
            movie->SetData(buf);
            movie->Write();
        }

        //display update info to console 
        if ((g->currentIteration % display_freq == 0) && (g->currentIteration > 1))
        {
            tprevious = tcurrent;
            tcurrent = clock();
            speed = (double)g->domainSize*display_freq / ((double)(tcurrent - tprevious) / CLOCKS_PER_SEC) / 1e6; //Mcells/s
            elapsed_time = (double)(tcurrent - tstart)/CLOCKS_PER_SEC ;
            printf("\rstep %d / %d, elapsed time %.2f s, speed: %.2f Mcells/s             ", g->currentIteration, g->nt-1 , elapsed_time, speed);
        }
            
    }
    printf("\nDone!\n");
    tcurrent = clock();
    speed = (double)g->domainSize*g->nt / ((double)(tcurrent - tstart) / CLOCKS_PER_SEC) / 1e6; //Mcells/s
    elapsed_time = (double)(tcurrent - tstart) / CLOCKS_PER_SEC;
    printf("Simulation stats: steps %d, elapsed time %.2f s, avg. speed: %.2f Mcells/s ", g->nt, elapsed_time, speed);



    if (MOVIE)
    {
        movie->End();
    }
    //retrieve all field data (final state)
   // CudaRetrieveAll(g, dg);
    CudaRetrieveField(g->detEx, dg->detEx, sizeof(float)*g->nt);
    CudaWriteTimeSeriesData("timeEx.f2d", g->detEx, g->nt);

    printf("\n");
    CudaFreeFields(dg);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(transfer_stream);
    exit(0);
}