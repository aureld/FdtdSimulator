// update.c : update equations for E and H
// Aurelien Duval 2014

#include "fdtd.h"

//update equation for H field
void updateH(Grid *g) 
{
	int mm;

	for (mm = 0; mm < SizeX - 1; mm++)
	{
		Hy(mm) =	Chyh(mm) * Hy(mm) +
					Chye(mm) * (Ez(mm + 1) - Ez(mm));
	}

	return;
}

//update equation for E field
void updateE(Grid *g)
{
	int mm;

	for (mm = 1; mm < SizeX - 1; mm++)
	{
		Ez(mm) =	Ceze(mm) * Ez(mm) +
					Cezh(mm) * (Hy(mm) - Hy(mm - 1));
	}

	return;
}


