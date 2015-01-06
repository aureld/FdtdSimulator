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
	Snapshot *snapshots;

	ALLOC_1D(g, 1, Grid);


	gridInit(g);
	ezIncInit(g);

	snapshotInit(g);
	snaphotInitStartTime(0);
	snaphotInitTemporalStride(5);

	//we create a tiff snapshot
	g->nbSnapshots = 1;
	ALLOC_1D(snapshots, g->nbSnapshots, Snapshot);
	snapshots[0] = snapshotSetType(TIFFIMG);
	snapshots[0].direction = XY;
	snapshots[0].field = EX;
	snapshots[0].slice = g->sizeZ / 2;
	snapshots[0].filename = "results/sim";
	snapshots[0].width = g->sizeX;
	snapshots[0].height = g->sizeY;


	/* time stepping */
	do_time_stepping(g, snapshots);

	return 0;

}

