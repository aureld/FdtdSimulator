// fdtdCore.c : FDTD core engine in C
// see http://www.eecs.wsu.edu/~schneidj
// Aurelien Duval 2014

#include <stdio.h>
#include "fdtd-alloc.h"
#include "fdtd-macros.h"
#include "fdtd-protos.h"


int main(int argc, char* argv[])
{
	Grid *g;
	ALLOC_1D(g, 1, Grid);


	gridInit(g, 3,3,3,100);
	ezIncInit(g);

	COW_MovieEngine *movie = new COW_MovieEngine;
	movie->SetFileName("test.avi");
	movie->SetFrameRate(30);
	movie->SetOutputSize(g->sizeX, g->sizeY);
	movie->Initialize(g->sizeX, g->sizeY);
	movie->StartMovieFFMPEG();
	

	/* time stepping */
	do_time_stepping(g, NULL, movie);

	movie->EndMovie();
	return 0;

}

