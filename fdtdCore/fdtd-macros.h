//fdtd-macros.h: Macros used for accessing arrays and structs
// Aurelien Duval 2014

#ifndef _FDTD_MACROS_H_
#define _FDTD_MACROS_H_

#include "fdtd-grid.h"

//Constants
#define M_PI 3.14159265358979323846 /* pi */
#define CHAR_BIT 8

//types
typedef enum { false, true } bool;
typedef enum { XY, XZ, YZ } Orientation;
typedef enum {F3D, PRINT, TIFFIMG} SnapshotType;
typedef void(*snapshotHeader)(Grid *g, Orientation direction, int slice);
typedef void(*snapshotBody)(Grid *g, float *field, int i, int j, int k);
typedef void(*snapshotRowDelim)(int i);
typedef void(*snapshotFooter)();

//used to define different types of snapshots - see snapshot.c
typedef struct {
	snapshotHeader header;
	snapshotBody body;
	snapshotRowDelim rowDelim;
	snapshotFooter footer;
} Snapshot;

//Array macros

__inline long idx(Grid *g, int i, int j, int k)
{
	return (long)i * g->sizeY * g->sizeZ + (long)j * g->sizeZ + (long)k;
}

__inline long idx2d(int dim1, int dim2, int i, int j)
{
	return (long)i * dim1 + (long)j;
}


#endif /*_FDTD_MACROS_H_*/