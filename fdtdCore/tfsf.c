// tfsf.c : TFSF input field code
// Aurelien Duval 2014

#include <math.h>
#include "fdtd.h"
#include "ezinc.h"

static int tfsfBoundary = 0;

//initializes the tfsf boundary plane at a constant (hardcoded) location
void tfsfInit(Grid *g)
{
	tfsfBoundary = 50; //position of the boundary plane
	ezIncInit(g);
	return;
}

//updates the tfsf source field
void tfsfUpdate(Grid *g)
{
	if (tfsfBoundary <= 0) {
		fprintf(stderr, "tfsfUpdate: uninitialized input, boundary location must be > 0.\n");
		exit(-1);
	}

	/*corrects E and H on each side of the boundary*/
	Hy(tfsfBoundary) -= ezInc(Time, 0.0) * Chye(tfsfBoundary);
	Ez(tfsfBoundary + 1) += ezInc(Time + 0.5, -0.5);

	return;
}
