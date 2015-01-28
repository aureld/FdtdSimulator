// stepping.c : performs FDTD time-stepping
// Aurelien Duval 2014

#include <stdio.h>
#include "fdtd-protos.h"
#include "fdtd-macros.h"
#include "MovieLib.h"

void do_time_stepping(Grid *g, COW_MovieEngine *movie)
{
	unsigned char *buf = new unsigned char[g->sizeX * g->sizeY * 3];
	for (g->time = 0; g->time < g->maxTime; g->time++)
	{
		//update fields
		updateH(g);
		updateE(g);

		// source
		g->ex[idx(g, g->sizeX / 2, g->sizeY / 2, g->sizeZ / 2)] += ezInc(g->time, 0);


		//write movie frame
		if (movie)
		{
			
			int pos = 0;
			int i;
			double maxcolors = 1.0;// ezInc(g->time, 0);
			double mincolors = 0.0;
			double normfact = 255.0 / (maxcolors - mincolors);
			for (int y = 0; y < g->sizeY; y++)
				for (int x = 0; x <g->sizeX; x++)
				{
				pos =3* ( y * g->sizeX + x);
				i = idx(g, x, y, g->sizeZ / 2);
				double val = (g->ex[i] - mincolors) * normfact;
				buf[pos]	 = val;
				buf[pos + 1] = val;
				buf[pos + 2] = val;
				}


			movie->SetData(buf, g->sizeX, g->sizeY);
			movie->WriteMovie();
		}

		//update console display
		printf("time: %d / %d\n", g->time, g->maxTime - 1);
	}
	//delete[] buf;
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