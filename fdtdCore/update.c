// update.c : update equations for E and H
// Aurelien Duval 2014

#include "fdtd-macros.h"


//update equation for H field
void updateH(Grid *g) 
{
	int mm,nn;

	if (Type == oneDGrid) {
		for (mm = 0; mm < SizeX - 1; mm++)
			Hy1(mm) = Chyh1(mm) * Hy1(mm) 
				+ Chye1(mm) * (Ez1(mm + 1) - Ez1(mm));
	}
	else {
		//Hx update
		for (mm = 0; mm < SizeX; mm++)
			for (nn = 0; nn < SizeY - 1; nn++)
				Hx(mm, nn) = Chxh(mm, nn) * Hx(mm, nn)
					- Chxe(mm, nn) * (Ez(mm, nn + 1) - Ez(mm, nn));
		//Hy update
		for (mm = 0; mm < SizeX - 1; mm++)
			for (nn = 0; nn < SizeY; nn++)
				Hx(mm, nn) = Chyh(mm, nn) * Hy(mm, nn)
				+ Chye(mm, nn) * (Ez(mm + 1, nn) - Ez(mm, nn));
	}
	return;
}

//update equation for E field
void updateE(Grid *g)
{
	int mm, nn;

	if (Type == oneDGrid) {
		for (mm = 1; mm < SizeX - 1; mm++)
			Ez1(mm) = Ceze1(mm) * Ez1(mm) 
			+ Cezh1(mm) * (Hy1(mm) - Hy1(mm - 1));
	}
	else {
		//Ez update
		for (mm = 1; mm < SizeX - 1; mm++)
			for (nn = 1; nn < SizeY-1; nn++)
				Ez(mm, nn) = Ceze(mm, nn) * Ez(mm, nn)
				+ Cezh(mm, nn) * ((Hy(mm, nn) - Hy(mm - 1, nn))
								- (Hx(mm,nn) - Hx(mm, nn - 1)));
	}
	return;
}


