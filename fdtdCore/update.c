// update.c : update equations for E and H
// Aurelien Duval 2014

#include <stdio.h>
#include "fdtd-macros.h"
#include "fdtd-protos.h"


//update equation for H field
void updateH(Grid *g) 
{
	if (g->type == threeDGrid) {
		updateHx(g);
		updateHy(g);
		updateHz(g);
	}
	else {
		fprintf(stderr, "UpdateH: unknown grid type.\n");
	}
	return;
}

void updateHx(Grid *g)
{
	int mm, nn, pp;
	for (mm = 0; mm < g->sizeX; mm++)
	{
		for (nn = 0; nn < g->sizeY-1; nn++)
		{
			for (pp = 0; pp < g->sizeZ-1; pp++)
			{
				g->hx[idx(g, mm, nn, pp)] = g->chxh[idx(g, mm, nn, pp)] * g->hx[idx(g, mm, nn, pp)]
					+ g->chxe[idx(g, mm, nn, pp)] * ((g->ey[idx(g, mm, nn, pp + 1)] - g->ey[idx(g, mm, nn, pp)])
													-(g->ez[idx(g, mm, nn + 1, pp)] - g->ez[idx(g, mm, nn, pp)]));
			}
		}
	}
}

void updateHy(Grid *g)
{
	int mm, nn, pp;
	for (mm = 0; mm < g->sizeX-1; mm++)
	{
		for (nn = 0; nn < g->sizeY; nn++)
		{
			for (pp = 0; pp < g->sizeZ-1; pp++)
			{
				g->hy[idx(g, mm, nn, pp)] = g->chyh[idx(g, mm, nn, pp)] * g->hy[idx(g, mm, nn, pp)]
					+ g->chye[idx(g, mm, nn, pp)] * ((g->ez[idx(g, mm + 1, nn, pp)] - g->ez[idx(g, mm, nn, pp)])
													-(g->ex[idx(g, mm, nn, pp + 1)] - g->ex[idx(g, mm, nn, pp)]));
			}
		}
	}
}

void updateHz(Grid *g)
{
	int mm, nn, pp;
	for (mm = 0; mm < g->sizeX-1; mm++)
	{
		for (nn = 0; nn < g->sizeY-1; nn++)
		{
			for (pp = 0; pp < g->sizeZ; pp++)
			{
				g->hz[idx(g, mm, nn, pp)] = g->chzh[idx(g, mm, nn, pp)] * g->hz[idx(g, mm, nn, pp)]
					+ g->chze[idx(g, mm, nn, pp)] * ((g->ex[idx(g, mm, nn + 1, pp)] - g->ex[idx(g, mm, nn, pp)])
													-(g->ey[idx(g, mm + 1, nn, pp)] - g->ey[idx(g, mm, nn, pp)]));
			}
		}
	}
}

//update equation for E field
void updateE(Grid *g)
{
	if (g->type == threeDGrid) {
		updateEx(g);
		updateEy(g);
		updateEz(g);
	}
	else {
		fprintf(stderr, "UpdateE: unknown grid type.\n");
	}
	return;
}

void updateEx(Grid *g)
{
	int mm, nn, pp;
	for (mm = 0; mm < g->sizeX - 1; mm++)
	{
		for (nn = 1; nn < g->sizeY - 1; nn++)
		{
			for (pp = 1; pp < g->sizeZ - 1; pp++)
			{
				g->ex[idx(g, mm, nn, pp)] = g->cexe[idx(g, mm, nn, pp)] * g->ex[idx(g, mm, nn, pp)]
					+ g->cexh[idx(g, mm, nn, pp)] * ((g->hz[idx(g, mm, nn, pp)] - g->hz[idx(g, mm, nn - 1, pp)])
					- (g->hy[idx(g, mm, nn, pp)] - g->hy[idx(g, mm, nn, pp - 1)]));
			}
		}
	}
}

void updateEy(Grid *g)
{
	int mm, nn, pp;
	for (mm = 1; mm < g->sizeX - 1; mm++)
	{
		for (nn = 0; nn < g->sizeY - 1; nn++)
		{
			for (pp = 1; pp < g->sizeZ - 1; pp++)
			{
				g->ey[idx(g, mm, nn, pp)] = g->ceye[idx(g, mm, nn, pp)] * g->ey[idx(g, mm, nn, pp)]
					+ g->ceyh[idx(g, mm, nn, pp)] * ((g->hx[idx(g, mm, nn, pp)] - g->hx[idx(g, mm, nn, pp - 1)])
					- (g->hz[idx(g, mm, nn, pp)] - g->hz[idx(g, mm - 1, nn, pp)]));
			}
		}
	}
}

void updateEz(Grid *g)
{
	int mm, nn, pp;
	for (mm = 1; mm < g->sizeX - 1; mm++)
	{
		for (nn = 1; nn < g->sizeY - 1; nn++)
		{
			for (pp = 0; pp < g->sizeZ - 1; pp++)
			{
				g->ez[idx(g, mm, nn, pp)] = g->ceze[idx(g, mm, nn, pp)] * g->ez[idx(g, mm, nn, pp)]
					+ g->cezh[idx(g, mm, nn, pp)] * ((g->hy[idx(g, mm, nn, pp)] - g->hy[idx(g, mm - 1, nn, pp)])
					- (g->hx[idx(g, mm, nn, pp)] - g->hx[idx(g, mm, nn - 1, pp)]));
			}
		}
	}
}




