//fdtd-alloc.h: memory allocation of the fields etc
// Aurelien Duval 2014

#ifndef _FDTD_ALLOC_H_
#define _FDTD_ALLOC_H_

#include <stdio.h>
#include <stdlib.h>


//Memory macros
#define ALLOC_1D(PNTR, NUM, TYPE)									\
	PNTR = (TYPE *)calloc(NUM, sizeof(TYPE));						\
	if (!PNTR) {													\
		perror("ALLOC_1D");											\
		fprintf(stderr, "Allocation failed for " #PNTR" \n");		\
		exit(-1);													\
		}

#define ALLOC_2D(PNTR, NUMX, NUMY, TYPE)							\
	PNTR = (TYPE *)calloc((NUMX) * (NUMY), sizeof(TYPE));			\
	if (!PNTR) {													\
		perror("ALLOC_2D");											\
		fprintf(stderr, "Allocation failed for " #PNTR" \n");		\
		exit(-1);													\
			}


#endif /*_FDTD_ALLOC_H_*/