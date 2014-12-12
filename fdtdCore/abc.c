// update.c : Absorbing boundary conditions
// Aurelien Duval 2014

#include "fdtd-alloc.h"
#include "fdtd-macros.h"


//macros 
#define Eyx0(N, P)	eyx0[(N) * (SizeZ)		+ (P)] // x left face, Ey tangential field
#define Ezx0(N, P)	ezx0[(N) * (SizeZ - 1)	+ (P)] // x left face, Ez tangential field
#define Eyx1(N, P)	eyx1[(N) * (SizeZ)		+ (P)] // x right face, Ey tangential field
#define Ezx1(N, P)	ezx1[(N) * (SizeZ - 1)	+ (P)] // x right face, Ez tangential field

#define Exy0(N, P)	exy0[(N) * (SizeZ)		+ (P)] // y left face, Ex tangential field
#define Ezy0(N, P)	ezy0[(N) * (SizeZ - 1)	+ (P)] // y left face, Ez tangential field
#define Exy1(N, P)	exy1[(N) * (SizeZ)		+ (P)] // y right face, Ex tangential field
#define Ezy1(N, P)	ezy1[(N) * (SizeZ - 1)	+ (P)] // y right face, Ez tangential field

#define Exz0(N, P)	exz0[(N) * (SizeY)		+ (P)] // z left face, Ex tangential field
#define Eyz0(N, P)	eyz0[(N) * (SizeY - 1)	+ (P)] // z left face, Ey tangential field
#define Exz1(N, P)	exz1[(N) * (SizeY)		+ (P)] // z right face, Ex tangential field
#define Eyz1(N, P)	eyz1[(N) * (SizeY - 1)	+ (P)] // z right face, Ey tangential field

static float abccoef = 0.0;
static float *eyx0, *eyx1, *ezx0, *ezx1;
static float *exy0, *exy1, *ezy0, *ezy1;
static float *exz0, *exz1, *eyz0, *eyz1;


//initializes the ABC (1st order diff equation)
void abcInit(Grid *g)
{
	abccoef = (Cdtds - 1.0) / (Cdtds + 1.0);

	//memory allocation
	ALLOC_2D(eyx0, SizeY - 1, SizeZ, float);
	ALLOC_2D(ezx0, SizeY, SizeZ - 1, float);
	ALLOC_2D(eyx1, SizeY - 1, SizeZ, float);
	ALLOC_2D(ezx1, SizeY, SizeZ - 1, float);
	
	ALLOC_2D(exy0, SizeY - 1, SizeZ, float);
	ALLOC_2D(ezy0, SizeY, SizeZ - 1, float);
	ALLOC_2D(exy1, SizeY - 1, SizeZ, float);
	ALLOC_2D(ezy1, SizeY, SizeZ - 1, float);

	ALLOC_2D(exz0, SizeY - 1, SizeZ, float);
	ALLOC_2D(eyz0, SizeY, SizeZ - 1, float);
	ALLOC_2D(exz1, SizeY - 1, SizeZ, float);
	ALLOC_2D(eyz1, SizeY, SizeZ - 1, float);

	return;
}

//apply the ABC to all 6 faces of the domain
void abc(Grid *g)
{
	int mm, nn, pp;

	if (abccoef == 0.0) {
		fprintf(stderr, "abc: uninitialized boundaries, call abcInit first.\n");
		exit(-1);
	}

	//x left boundary
	mm = 0;
	for (nn = 0; nn < SizeY - 1; nn++)
		for (pp = 0; pp < SizeZ; pp++)
		{
			Ey(mm, nn, pp) = Eyx0(nn, pp)
							+ abccoef * (Ey(mm + 1, nn, pp) - Ey(mm, nn, pp));
			Eyx0(nn, pp) = Ey(mm + 1, nn, pp);
		}
	for (nn = 0; nn < SizeY; nn++)
		for (pp = 0; pp < SizeZ - 1; pp++)
		{
			Ez(mm, nn, pp) = Ezx0(nn, pp)
				+ abccoef * (Ez(mm + 1, nn, pp) - Ez(mm, nn, pp));
			Ezx0(nn, pp) = Ez(mm + 1, nn, pp);
		}
		
	//x right boundary
	mm = SizeX - 1;
	for (nn = 0; nn < SizeY - 1; nn++)
		for (pp = 0; pp < SizeZ; pp++)
		{
			Ey(mm, nn, pp) = Eyx1(nn, pp)
				+ abccoef * (Ey(mm - 1, nn, pp) - Ey(mm, nn, pp));
			Eyx1(nn, pp) = Ey(mm - 1, nn, pp);
		}
	for (nn = 0; nn < SizeY; nn++)
		for (pp = 0; pp < SizeZ - 1; pp++)
		{
			Ez(mm, nn, pp) = Ezx1(nn, pp)
				+ abccoef * (Ez(mm - 1, nn, pp) - Ez(mm, nn, pp));
			Ezx1(nn, pp) = Ez(mm - 1, nn, pp);
		}

	//y left boundary
	nn = 0;
	for (mm = 0; mm < SizeX - 1; mm++)
		for (pp = 0; pp < SizeZ; pp++)
		{
			Ex(mm, nn, pp) = Exy0(nn, pp)
				+ abccoef * (Ex(mm, nn + 1, pp) - Ex(mm, nn, pp));
			Exy0(nn, pp) = Ex(mm, nn + 1, pp);
		}
	for (mm = 0; mm < SizeX; mm++)
		for (pp = 0; pp < SizeZ - 1; pp++)
		{
			Ez(mm, nn, pp) = Ezy0(nn, pp)
				+ abccoef * (Ez(mm, nn + 1, pp) - Ez(mm, nn, pp));
			Ezy0(nn, pp) = Ez(mm, nn + 1, pp);
		}
		
	//y right boundary
	nn = SizeY - 1;
	for (mm = 0; mm < SizeX - 1; mm++)
		for (pp = 0; pp < SizeZ; pp++)
		{
			Ex(mm, nn, pp) = Exy1(nn, pp)
				+ abccoef * (Ex(mm, nn - 1, pp) - Ex(mm, nn, pp));
			Exy1(nn, pp) = Ex(mm, nn - 1, pp);
		}
	for (mm = 0; mm < SizeX; mm++)
		for (pp = 0; pp < SizeZ - 1; pp++)
		{
			Ez(mm, nn, pp) = Ezy1(nn, pp)
				+ abccoef * (Ez(mm, nn - 1, pp) - Ez(mm, nn, pp));
			Ezy1(nn, pp) = Ez(mm, nn - 1, pp);
		}

	//z left boundary
	pp = 0;
	for (mm = 0; mm < SizeX - 1; mm++)
		for (nn = 0; nn < SizeY; nn++)
		{
			Ex(mm, nn, pp) = Exz0(nn, pp)
				+ abccoef * (Ex(mm, nn, pp + 1) - Ex(mm, nn, pp));
			Exz0(nn, pp) = Ex(mm, nn, pp + 1);
		}
	for (mm = 0; mm < SizeX; mm++)
		for (nn = 0; nn < SizeY - 1; nn++)
		{
			Ey(mm, nn, pp) = Eyz0(nn, pp)
				+ abccoef * (Ey(mm, nn, pp + 1) - Ey(mm, nn, pp));
			Eyz0(nn, pp) = Ey(mm, nn, pp + 1);
		}

	//z right boundary
	pp = SizeZ - 1;
	for (mm = 0; mm < SizeX - 1; mm++)
		for (nn = 0; nn < SizeY; nn++)
		{
		Ex(mm, nn, pp) = Exz1(nn, pp)
			+ abccoef * (Ex(mm, nn, pp - 1) - Ex(mm, nn, pp));
		Exz1(nn, pp) = Ex(mm, nn, pp - 1);
		}
	for (mm = 0; mm < SizeX; mm++)
		for (nn = 0; nn < SizeY - 1; nn++)
		{
		Ey(mm, nn, pp) = Eyz1(nn, pp)
			+ abccoef * (Ey(mm, nn, pp - 1) - Ey(mm, nn, pp));
		Eyz1(nn, pp) = Ey(mm, nn, pp - 1);
		}
	return;
}