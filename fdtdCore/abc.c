// update.c : Absorbing boundary conditions
// Aurelien Duval 2014

#include "fdtd.h"
#include <math.h>

static int initDone = 0;
static double *ezOldLeft1;
static double *ezOldLeft2;
static double *ezOldRight1;
static double *ezOldRight2;
static double *abcCoefLeft;
static double *abcCoefRight;

//initializes the ABC (2nd order diff equation or Mur ABC)
void abcInit(Grid *g)
{
	double temp1, temp2;

	ALLOC_1D(ezOldLeft1, 3, double);
	ALLOC_1D(ezOldLeft2, 3, double);
	ALLOC_1D(ezOldRight1, 3, double);
	ALLOC_1D(ezOldRight2, 3, double);
	ALLOC_1D(abcCoefLeft, 3, double);
	ALLOC_1D(abcCoefRight, 3, double);


	/* left boundary coefficient */
	temp1 = sqrt(Cezh(0) * Chye(0));
	temp2 = 1.0 / temp1 + 2.0 + temp1;
	abcCoefLeft[0] = -(1.0 / temp1 - 2.0 + temp1) / temp2;
	abcCoefLeft[1] = -2.0 * (temp1 - 1.0 / temp1) / temp2;
	abcCoefLeft[2] =  4.0 * (temp1 + 1.0 / temp1) / temp2;


	/* right boundary coefficient */
	temp1 = sqrt(Cezh(SizeX -1) * Chye(SizeX -2));
	temp2 = 1.0 / temp1 + 2.0 + temp1;
	abcCoefRight[0] = -(1.0 / temp1 - 2.0 + temp1) / temp2;
	abcCoefRight[1] = -2.0 * (temp1 - 1.0 / temp1) / temp2;
	abcCoefRight[2] = 4.0 * (temp1 + 1.0 / temp1) / temp2;

	initDone = 1;
	return;
}

//apply the ABC
void abc(Grid *g)
{
	int mm ;

	if (!initDone) {
		fprintf(stderr, "abc: uninitialized boundaries, call abcInit first.\n");
		exit(-1);
	}

	/* left boundary */
	Ez(0) =		abcCoefLeft[0] * (Ez(2) + ezOldLeft2[0])
			+	abcCoefLeft[1] * (ezOldLeft1[0] + ezOldLeft1[2] - Ez(1) - ezOldLeft2[1])
			+	abcCoefLeft[2] *  ezOldLeft1[1] - ezOldLeft2[2];
								;

	/* right boundary */
	Ez(SizeX-1) =	abcCoefRight[0] * (Ez(SizeX -3) + ezOldRight2[0])
				+	abcCoefRight[1] * (ezOldRight1[0] + ezOldRight1[2] - Ez(SizeX-2) - ezOldRight2[1])
				+	abcCoefRight[2] * ezOldRight1[1] - ezOldRight2[2];

	/* update old fields */
	for (mm = 0; mm < 3; mm++)
	{
		ezOldLeft2[mm] = ezOldLeft1[mm];
		ezOldLeft1[mm] = Ez(mm);

		ezOldRight2[mm] = ezOldRight1[mm];
		ezOldRight1[mm] = Ez(SizeX -1 - mm);
	}

	return;
}