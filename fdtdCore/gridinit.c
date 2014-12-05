// gridinit.c : Grid initialization code
// Aurelien Duval 2014

#include "fdtd.h"


#define EPSR		9.0		//material epsilon

//initializes the grid
void gridInit(Grid *g)
{
	double imp0 = 377.0;	//cross impedance of free-space
	int mm;

	SizeX = 200;			//fdtd domain size
	MaxTime = 1000;			//simulation duration
	Cdtds = 1.0;			//courant number

	//memory allocation
	ALLOC_1D(g->ez, SizeX, double);
	ALLOC_1D(g->ceze, SizeX, double);
	ALLOC_1D(g->cezh, SizeX, double);

	ALLOC_1D(g->hy, SizeX-1, double);
	ALLOC_1D(g->chyh, SizeX - 1, double);
	ALLOC_1D(g->chye, SizeX - 1, double);

	/* E field update coefficients */
	for (mm = 0; mm < SizeX; mm++)
	{
		if (mm < 100) {
			Ceze(mm) = 1.0;
			Cezh(mm) = imp0;
		}
		else {
			Ceze(mm) = 1.0;
			Cezh(mm) = imp0 / EPSR;
		}
	}

	/* H field update coefficients */
	for (mm = 0; mm < SizeX - 1; mm++)
	{
			Chyh(mm) = 1.0;
			Chye(mm) = 1.0 / imp0;
	}

	return;
}
