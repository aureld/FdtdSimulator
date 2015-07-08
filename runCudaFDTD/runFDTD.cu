// runFDTD.cu : FDTD core engine in C, CUDA variant
// see http://www.eecs.wsu.edu/~schneidj
// Aurelien Duval 2015

#include <stdio.h>
#include "Cuda_fdtd_macros.h"
#include "Cuda_fdtd_alloc.h"
#include "fdtd-alloc.h"
#include "Cuda_fdtd-protos.h"
#include "MovieLib.h"


int main(int argc, char* argv[])
{
	Grid *g;

	ALLOC_1D(g, 1, Grid);


	Cuda_gridInit(g, 64, 64, 64, 5000);
	
	Movie *movie = new Movie();
	movie->SetFileName("testCuda.avi");
	movie->Initialize(g->sizeX, g->sizeY);
	movie->Start();

	/* time stepping */
	Cuda_do_time_stepping(g, movie);
	movie->End();
	return 0;

}