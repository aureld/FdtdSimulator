//fdtd-macros.h: Macros used for accessing arrays and structs
// Aurelien Duval 2014
#pragma once
#ifndef _FDTD_MACROS_H_
#define _FDTD_MACROS_H_

#include "fdtd-structs.h"
#include "tiffio.h"
//Constants

#define CHAR_BIT 8

#ifdef _MSC_VER
#define M_PI 3.14159265358979323846 /* pi */
#ifndef __cplusplus
#define inline __inline
#endif
#endif /* _MSC_VER */

//types
typedef enum { false, true } bool;
typedef enum { XY, XZ, YZ } Orientation;
typedef enum {F3D, PRINT, TIFFIMG} SnapshotType;
typedef enum {EX, EY, EZ, HX, HY, HZ} FieldType;
typedef void(*snapshotHeader)(Grid *g, Orientation direction, int slice);
typedef void(*snapshotBody)(Grid *g, float *field, int i, int j, int k, int indexer);
typedef void(*snapshotRowDelim)(int i);
typedef void(*snapshotFooter)();

//used to define different types of snapshots - see snapshot.c
typedef struct {
	FieldType field; // field being captured
	int slice; // position of the slice along the orthogonal axis
	Orientation direction; //orientation of the plane of capture
	int width; //width of the snapshot
	int height; //height of the snapshot
	//file data
	char *filename; //absolute path + filename
	FILE *snapfile; //file handle
	TIFF *tifffile; //file handle (tiff images only)
	//these are relative to the snapshot function callbacks
	snapshotHeader header;
	snapshotBody body;
	snapshotRowDelim rowDelim;
	snapshotFooter footer;
} Snapshot;

//Array macros
#define idx(g,i,j,k) ((long)(i) * (g->sizeY) * (g->sizeZ) + (long)(j) * (g->sizeZ) + (long)(k))
#define idx2d(dim1,i,j) ((long)i()) * (dim1) + (long)(j))



#endif /*_FDTD_MACROS_H_*/
