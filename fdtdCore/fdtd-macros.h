//fdtd-macros.h: Macros used for accessing arrays and structs
// Aurelien Duval 2014

#ifndef _FDTD_MACROS_H_
#define _FDTD_MACROS_H_

#include "fdtd-grid.h"

//Constants
# define M_PI 3.14159265358979323846 /* pi */

//Array macros

__inline long idx(Grid *g, int i, int j, int k)
{
	return (long)i * g->sizeY * g->sizeZ + (long)j * g->sizeZ + (long)k;
}



#endif /*_FDTD_MACROS_H_*/