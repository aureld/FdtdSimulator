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

	gridInit(g);
	//abcInit(g);
	ezIncInit(g);

	snapshotInit(g);
	snaphotInitStartTime(0);
	snaphotInitTemporalStride(5);
	Snapshot f3dSnap = snapshotSetType(F3D);
	Snapshot tiffSnap = snapshotSetType(TIFFIMG);

	/* time stepping */
	for (g->time = 0; g->time < g->maxTime; g->time++)
	{
		updateH(g);
		updateE(g);
		g->ex[idx(g, g->sizeX / 2, g->sizeY / 2, g->sizeZ / 2)] += ezInc(g->time, 0); // source
		//abc(g);
		
		snapshot(g, g->ex, 1, 0, tiffSnap);
		//snapshot(g, g->ex, 1, 0, f3dSnap);

		printf("time: %d / %d\n", g->time, g->maxTime-1);
	}

	return 0;

}

