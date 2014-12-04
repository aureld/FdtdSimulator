// ezincharm.c : input field with harmonic time dependance
// Aurelien Duval 2014

#include "ezinc.h"

static double ppw = 0; //points per wavelength
static double cdtds;

//initializes the input field with hardcoded values
void ezIncInit(Grid *g)
{
	cdtds = Cdtds;
	ppw = 40;
}

//calculates the input field at specified time and location
double ezInc(double time, double location)
{
	if (ppw <= 0) {
		fprintf(stderr, "ezincharm: uninitialized input, ppw must be > 0.\n");
		exit(-1);
	}

	return sin(2.0 * M_PI / ppw * (cdtds * time - location));
}