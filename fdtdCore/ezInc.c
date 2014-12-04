// ezInc.c : input field code (Gaussian pulse)
// Aurelien Duval 2014


#include "ezinc.h"

static double delay;
static double width = 0;
static double cdtds;

//prompts for delay and width of the pulse
void ezIncInit(Grid *g)
{
	cdtds = Cdtds;
	delay = 1;
	width = 10;

	return;
}

//returns the source field value for current position and location
double ezInc(double time, double location)
{
	if (width <= 0) {
		fprintf(stderr, "ezInc: uninitialized input, width must be > 0.\n");
		exit(-1);
	}
	return exp(-pow((time - delay - location/cdtds) / width, 2));
}
