// update.c : Absorbing boundary conditions
// Aurelien Duval 2014

#include "fdtd.h"
#include <math.h>

static int initDone = 0;
static double ezOldLeft = 0.0;
static double ezOldRight = 0.0;
static double abcCoefLeft;
static double abcCoeRight;

//initializes the ABC (1st order diff equation)
void abcInit(Grid *g)
{
	double temp;

	/* left boundary coefficient */
	temp = sqrt(Cezh(0) * Chye(0));
	abcCoefLeft = (temp - 1.0) / (temp + 1.0);

	/* right boundary coefficient */
	temp = sqrt(Cezh(SizeX -1) * Chye(SizeX -2));
	abcCoeRight = (temp - 1.0) / (temp + 1.0);

	initDone = 1;
	return;
}

//apply the ABC -- left side only
void abc(Grid *g)
{
	if (!initDone) {
		fprintf(stderr, "abc: uninitialized boundaries, call abcInit first.\n");
		exit(-1);
	}

	/* left boundary */
	Ez(0) =			ezOldLeft + 
					abcCoefLeft * (Ez(1) - Ez(0));
	ezOldLeft = Ez(1);

	/* right boundary */
	Ez(SizeX -1) =	ezOldRight + 
					abcCoeRight * (Ez(SizeX -2) - Ez(SizeX - 1));
	ezOldRight = Ez(SizeX - 2);

	return;
}