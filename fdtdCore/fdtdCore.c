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
	ezIncInit(g);
	snapshotInit(g);

	/* time stepping */
	for (Time = 0; Time < MaxTime; Time++)
	{
		updateH(g);
		updateE(g);
		Ez(SizeX / 2, SizeY / 2) = ezInc(Time, 0.0); // source
		snapshot(g);
	}

	return 0;

}

