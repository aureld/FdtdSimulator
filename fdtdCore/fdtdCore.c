// fdtdCore.c : FDTD core engine in C
// see http://www.eecs.wsu.edu/~schneidj
// Aurelien Duval 2014

#include "fdtd-alloc.h"
#include "fdtd-macros.h"
#include "fdtd-protos.h"
#include "ezinc.h"

int main(int argc, char* argv[])
{
	Grid *g;

	ALLOC_1D(g, 1, Grid);

	gridInit(g);
	//abcInit(g);
	ezIncInit(g);
	snapshotInit(g);

	/* time stepping */
	for (Time = 0; Time < MaxTime; Time++)
	{
		updateH(g);
		updateE(g);
		g->ez[idx(g, g->sizeX / 2, g->sizeY / 2, g->sizeZ / 2)] += ezInc(Time, 0.0); // source
		//abc(g);
		//snapshot(g);
		print(g, 1, 0);
	}

	return 0;

}

