// runFDTD.c : FDTD core engine in C
// see http://www.eecs.wsu.edu/~schneidj
// Aurelien Duval 2014

#include <stdio.h>
#include "fdtd-alloc.h"
#include "fdtd-macros.h"
#include "fdtd-protos.h"
#include "MovieLib.h"


int main(int argc, char* argv[])
{
	Grid *g;
	ALLOC_1D(g, 1, Grid);


	gridInit(g, 64,64,64,500);
	ezIncInit(g);

	Movie *movie = new Movie();
	movie->SetFileName("test.avi");
	movie->Initialize(g->sizeX, g->sizeY);
	movie->Start();
	
	/* time stepping */
	do_time_stepping(g, movie);
	movie->End();
	return 0;

}

