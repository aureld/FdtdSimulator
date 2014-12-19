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
	float coefmul, coefdiv;
	
	g->type = threeDGrid;
	g->sizeX = 256;							//fdtd domain size X
	g->sizeY = 256;							//fdtd domain size Y
	g->sizeZ = 6;							//fdtd domain size Z
	g->maxTime = 500;						//simulation duration
	g->cdtds = (float)(1.0 / sqrt(3.0));			//courant number for 3D

	//memory allocation
	ALLOC_3D(g->hx,		g->sizeX,	g->sizeY,	g->sizeZ,	float);
	ALLOC_3D(g->chxh,	g->sizeX,	g->sizeY,	g->sizeZ,	float);
	ALLOC_3D(g->chxe,	g->sizeX,	g->sizeY,	g->sizeZ,	float);
	ALLOC_3D(g->hy,		g->sizeX,	g->sizeY,	g->sizeZ,	float);
	ALLOC_3D(g->chyh,	g->sizeX,	g->sizeY,	g->sizeZ,	float);
	ALLOC_3D(g->chye,	g->sizeX,	g->sizeY,	g->sizeZ,	float);
	ALLOC_3D(g->hz,		g->sizeX,	g->sizeY,	g->sizeZ,	float);
	ALLOC_3D(g->chzh,	g->sizeX,	g->sizeY,	g->sizeZ,	float);
	ALLOC_3D(g->chze,	g->sizeX,	g->sizeY,	g->sizeZ,	float);
												
	ALLOC_3D(g->ex,		g->sizeX,	g->sizeY,	g->sizeZ,	float);
	ALLOC_3D(g->cexe,	g->sizeX,	g->sizeY,	g->sizeZ,	float);
	ALLOC_3D(g->cexh,	g->sizeX,	g->sizeY,	g->sizeZ,	float);
	ALLOC_3D(g->ey,		g->sizeX,	g->sizeY ,	g->sizeZ,	float);
	ALLOC_3D(g->ceye,	g->sizeX,	g->sizeY ,	g->sizeZ,	float);
	ALLOC_3D(g->ceyh,	g->sizeX,	g->sizeY ,	g->sizeZ,	float);
	ALLOC_3D(g->ez,		g->sizeX,	g->sizeY ,	g->sizeZ,	float);
	ALLOC_3D(g->ceze,	g->sizeX,	g->sizeY,	g->sizeZ,	float);
	ALLOC_3D(g->cezh,	g->sizeX,	g->sizeY,	g->sizeZ,	float);
	

	/* Ex field update coefficients */
	coefmul = g->cdtds * imp0;
	coefdiv = g->cdtds / imp0;

	for (mm = 0; mm < g->sizeX; mm++)
	{
		for (nn = 0; nn < g->sizeY; nn++)
		{
			for (pp = 0; pp < g->sizeZ; pp++)
			{
				i = idx(g, mm, nn, pp);

				g->cexe[i] = 1.0;
				g->cexh[i] = coefmul;
				g->ceye[i] = 1.0;
				g->ceyh[i] = coefmul;
				g->ceze[i] = 1.0;
				g->cezh[i] = coefmul;

				g->chxh[i] = 1.0;
				g->chxe[i] = coefdiv;
				g->chyh[i] = 1.0;
				g->chye[i] = coefdiv;
				g->chzh[i] = 1.0;
				g->chze[i] = coefdiv;
			}
		}
	}
	return;
}
