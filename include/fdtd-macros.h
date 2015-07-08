//fdtd-macros.h: Macros used for accessing arrays and structs
// Aurelien Duval 2014
#pragma once
#ifndef _FDTD_MACROS_H_
#define _FDTD_MACROS_H_

#include "fdtd-structs.h"
//Constants

#define CHAR_BIT 8

#ifdef _MSC_VER
#define M_PI 3.14159265358979323846 /* pi */
#ifndef __cplusplus
#define inline __inline
#endif
#endif /* _MSC_VER */

//types
typedef enum { XY, XZ, YZ } Orientation;
typedef enum {F3D, PRINT, TIFFIMG} SnapshotType;
typedef enum {EX, EY, EZ, HX, HY, HZ} FieldType;



//Array macros
#ifdef _DEBUG
inline long idx(Grid *g, int i, int j, int k) { return ((long)(i)* (g->sizeY) * (g->sizeZ) + (long)(j)* (g->sizeZ) + (long)(k)); }
#else
#define idx(g, i, j, k) ((long)(i) * (g->sizeY) * (g->sizeZ) + (long)(j) * (g->sizeZ) + (long)(k))
#endif
#define idx2d(dim1,i,j) ((long)i()) * (dim1) + (long)(j))



#endif /*_FDTD_MACROS_H_*/
