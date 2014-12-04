//fdtd.h: general definition pertinent to FDTD core
// Aurelien Duval 2014

#ifndef _FDTD_H_
#define _FDTD_H_

#include <stdio.h>
#include <stdlib.h>

# define M_PI 3.14159265358979323846 /* pi */

//FDTD grid containing all field cells and time data
struct Grid {
	double		*ez;		//Ez field pointer (1D)
	double		*ceze;		//Ez update coeff (E comp)
	double		*cezh;		//Ez update coeff (H comp)
	double		*hy;		//Hy field pointer (1D)
	double		*chyh;		//Hy update coeff (H comp)
	double		*chye;		//Hy update coeff (E comp)
	int			sizeX;		//grid size
	int			time;		//current timestep
	int			maxTime;	//final timestep
	double		cdtds;		//courant number
};

typedef struct Grid Grid;

//Memory macros
#define ALLOC_1D(PNTR, NUM, TYPE)									\
	PNTR = (TYPE *)calloc(NUM, sizeof(TYPE));						\
	if (!PNTR) {													\
		perror("ALLOC_1D");											\
		fprintf(stderr, "Allocation failed for " #PNTR" \n");		\
		exit(-1);													\
	}

//Array macros
//the notation is dangerous, to be refactored later (classes in C++?)
#define Hy(MM)		g->hy[MM]
#define Chyh(MM)	g->chyh[MM]
#define Chye(MM)	g->chye[MM]

#define Ez(MM)		g->ez[MM]
#define Ceze(MM)	g->ceze[MM]
#define Cezh(MM)	g->cezh[MM]

#define SizeX		g->sizeX
#define Time		g->time
#define	MaxTime		g->maxTime
#define Cdtds		g->cdtds

//prototypes
void gridInit(Grid *g);
void updateH(Grid *g);
void updateE(Grid *g);

void abcInit(Grid *g);
void abc(Grid *g);

void tfsfInit(Grid *g);
void tfsfUpdate(Grid *g);

void snapshotInit(Grid *g);
void snapshot(Grid *g);


#endif /*_FDTD_H_*/