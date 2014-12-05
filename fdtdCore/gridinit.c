// gridinit.c : Grid initialization code
// Aurelien Duval 2014

#include <math.h>
#include "fdtd-macros.h"
#include "fdtd-alloc.h"

//initializes the grid
void gridInit(Grid *g)
{
	double imp0 = 377.0;	//cross impedance of free-space
	int mm,nn;
	
	Type = tmZGrid;
	SizeX = 101;			//fdtd domain size X
	SizeY = 81;			//fdtd domain size Y
	MaxTime = 300;			//simulation duration
	Cdtds = 1.0 / sqrt(2.0);			//courant number for 2D

	//memory allocation
	ALLOC_2D(g->hx,		SizeX,		SizeY - 1,	double);
	ALLOC_2D(g->chxh,	SizeX,		SizeY - 1,	double);
	ALLOC_2D(g->chxe,	SizeX,		SizeY - 1,	double);
	ALLOC_2D(g->hy,		SizeX - 1,	SizeY,		double);
	ALLOC_2D(g->chyh,	SizeX - 1,	SizeY,		double);
	ALLOC_2D(g->chye,	SizeX - 1,	SizeY,		double);
	ALLOC_2D(g->ez,		SizeX,		SizeY,		double);
	ALLOC_2D(g->ceze,	SizeX,		SizeY,		double);
	ALLOC_2D(g->cezh,	SizeX,		SizeY,		double);
	

	/* Ez field update coefficients */
	for (mm = 0; mm < SizeX; mm++)
		for (nn = 0; nn < SizeY; nn++)
		{
		Ceze(mm, nn) = 1.0;
		Cezh(mm, nn) = Cdtds * imp0;
		}

	/* Hx field update coefficients */
	for (mm = 0; mm < SizeX; mm++)
		for (nn = 0; nn < SizeY - 1; nn++)
		{
				Chxh(mm,nn) = 1.0;
				Chxe(mm,nn) = Cdtds / imp0;
		}

	/* Hy field update coefficients */
	for (mm = 0; mm < SizeX - 1; mm++)
		for (nn = 0; nn < SizeY; nn++)
		{
		Chyh(mm, nn) = 1.0;
		Chye(mm, nn) = Cdtds / imp0;
		}

	return;
}
