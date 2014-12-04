// update.c : Absorbing boundary conditions
// Aurelien Duval 2014

#include "fdtd.h"

//initializes the ABC
void abcInit(Grid *g)
{
	return;
}

//apply the ABC -- left side only
void abc(Grid *g)
{
	Ez(0) = Ez(1);

	return;
}