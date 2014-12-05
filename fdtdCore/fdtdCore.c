// fdtdCore.c : FDTD core engine in C
// see http://www.eecs.wsu.edu/~schneidj
// Aurelien Duval 2014

#include "fdtd.h"
#include "ezinc.h"

int main(int argc, char* argv[])
{
	Grid *g;

	ALLOC_1D(g, 1, Grid);

	gridInit(g);
	abcInit(g);
	tfsfInit(g);
	snapshotInit(g);

	/* time stepping */
	for (Time = 0; Time < MaxTime; Time++)
	{
		updateH(g);
		tfsfUpdate(g);
		updateE(g);
		abc(g);
		snapshot(g);
	}

	return 0;

}

