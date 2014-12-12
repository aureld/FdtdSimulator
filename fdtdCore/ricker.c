// ricker.c : Ricker wavelet source
// Aurelien Duval 2014

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fdtd-protos.h"
#include "fdtd-macros.h"


static double ppw = 0; //points per wavelength
static double cdtds;

//initializes the input field with hardcoded values
void ezIncInit(Grid *g)
{
	cdtds = g->cdtds;
	ppw = 20;
}

//calculates the input field at specified time and location
float ezInc(int time, int location)
{
	double arg;

	if (ppw <= 0) {
		fprintf(stderr, "ricker: uninitialized input, ppw must be > 0.\n");
		exit(-1);
	}

	arg = M_PI * ((cdtds * time - location) / ppw - 1.0);
	arg = arg * arg;

	return (float)((1.0 - 2.0 * arg) * exp(-arg));
}