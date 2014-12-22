// stepping.c : performs FDTD time-stepping
// Aurelien Duval 2014

#include <stdio.h>
#include "fdtd-protos.h"
#include "fdtd-macros.h"

void do_time_stepping(Grid *g, Snapshot *snapshots)
{
	for (g->time = 0; g->time < g->maxTime; g->time++)
	{
		//update fields
		updateH(g);
		updateE(g);

		// source
		g->ex[idx(g, g->sizeX / 2, g->sizeY / 2, g->sizeZ / 2)] += ezInc(g->time, 0);

		//capture snapshots
		if (snapshots)
		{
			int i;
			for (i = 0; i < g->nbSnapshots; i++)
				snapshot(g, field(g, snapshots[i].field), snapshots[i].slice, snapshots[i].direction, snapshots[i]);
		}

		//update console display
		printf("time: %d / %d\n", g->time, g->maxTime - 1);
	}
}

float *field(Grid *g, FieldType f)
{
	switch (f)
	{
	case EX: return g->ex; break;
	case EY: return g->ey; break;
	case EZ: return g->ez; break;
	case HX: return g->hx; break;
	case HY: return g->hy; break;
	case HZ: return g->hz; break;
	}
}