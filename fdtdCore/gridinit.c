// gridinit.c : Grid initialization code
// Aurelien Duval 2014

#include <math.h>
#include "fdtd-macros.h"
#include "fdtd-alloc.h"

//initializes the grid
void gridInit(Grid *g)
{
	float imp0 = 377.0;				//cross impedance of free-space
	int mm,nn,pp;
	long i;
	
	Type = threeDGrid;
	SizeX = 5;							//fdtd domain size X
	SizeY = 5;							//fdtd domain size Y
	SizeZ = 5;							//fdtd domain size Z
	MaxTime = 10;						//simulation duration
	Cdtds = 1.0 / sqrt(3.0);			//courant number for 3D

	//memory allocation
	ALLOC_3D(g->hx,		SizeX,	SizeY,	SizeZ,	float);
	ALLOC_3D(g->chxh,	SizeX,	SizeY,	SizeZ,	float);
	ALLOC_3D(g->chxe,	SizeX,	SizeY,	SizeZ,	float);
	ALLOC_3D(g->hy,		SizeX,	SizeY,	SizeZ,	float);
	ALLOC_3D(g->chyh,	SizeX,	SizeY,	SizeZ,	float);
	ALLOC_3D(g->chye,	SizeX,	SizeY,	SizeZ,	float);
	ALLOC_3D(g->hz,		SizeX,	SizeY,	SizeZ,	float);
	ALLOC_3D(g->chzh,	SizeX,	SizeY,	SizeZ,	float);
	ALLOC_3D(g->chze,	SizeX,	SizeY,	SizeZ,	float);

	ALLOC_3D(g->ex,		SizeX,	SizeY,	SizeZ,	float);
	ALLOC_3D(g->cexe,	SizeX,	SizeY,	SizeZ,	float);
	ALLOC_3D(g->cexh,	SizeX,	SizeY,	SizeZ,	float);
	ALLOC_3D(g->ey,		SizeX,	SizeY ,	SizeZ,	float);
	ALLOC_3D(g->ceye,	SizeX,	SizeY ,	SizeZ,	float);
	ALLOC_3D(g->ceyh,	SizeX,	SizeY ,	SizeZ,	float);
	ALLOC_3D(g->ez,		SizeX,	SizeY ,	SizeZ,	float);
	ALLOC_3D(g->ceze,	SizeX,	SizeY,	SizeZ,	float);
	ALLOC_3D(g->cezh,	SizeX,	SizeY,	SizeZ,	float);
	

	/* Ex field update coefficients */
	for (mm = 0; mm < g->sizeX; mm++)
	{
		for (nn = 0; nn < g->sizeY; nn++)
		{
			for (pp = 0; pp < g->sizeZ; pp++)
			{
				i = idx(g, mm, nn, pp);
				g->cexe[i] = 1.0;
				g->cexh[i] = Cdtds * imp0;
				g->ceye[i] = 1.0;
				g->ceyh[i] = Cdtds * imp0;
				g->ceze[i] = 1.0;
				g->cezh[i] = Cdtds * imp0;

				g->chxh[i] = 1.0;
				g->chxe[i] = Cdtds / imp0;
				g->chyh[i] = 1.0;
				g->chye[i] = Cdtds / imp0;
				g->chzh[i] = 1.0;
				g->chze[i] = Cdtds / imp0;
			}
		}
	}
	return;
}
