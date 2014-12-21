// snapshot.c : takes snapshots at time intervals of the grid
// Aurelien Duval 2014

#include <stdio.h>
#include <tiffio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fdtd-macros.h"
#include "fdtd-protos.h"

static int temporalStride = -2;
static int startTime;
static int frameX = 0;
static char basename[80] = "results/sim";
static char filename[100] = "";
static FILE* snapfile;
static TIFF* tif;
static int width, height;
static bool initDone = false;
static float *buf = NULL;

//checks if we are at a timestep where we should get a snapshot
bool strideConditionsMet(Grid *g)
{
	return (g->time >= startTime && (g->time - startTime) % temporalStride == 0);
}

//checks if the snapshots have been initialized (using a call to snapshotInit)
bool isInitialized() {
	if (!initDone) {
		fprintf(stderr, "snapshot: uninitialized snapshots.\n");
		exit(-1);
	}
	return true;
}

//initializes the snapshots with hardcoded values
void snapshotInit(Grid *g)
{
	snapfile = NULL;
	temporalStride = 10;
	startTime = 0;
	initDone = true;
	return;
}

//initializes the start time (default 0)
void snaphotInitStartTime(int start)
{
	if (start >= 0)
	{
		startTime = start;
		initDone = true;
	}
	else
	{
		fprintf(stderr, "snaphotInitStartTime: start time should be >= 0.\n");
		exit(-1);
	}
}

//initializes the start time (default 0)
void snaphotInitTemporalStride(int stride)
{
	if (stride > 0)
	{
		temporalStride = stride;
		initDone = true;
	}
	else
	{
		fprintf(stderr, "snaphotInitTemporalStride: stride should be > 0.\n");
		exit(-1);
	}
}

//returns a struct with snapshot function pointers for the type
Snapshot snapshotSetType(SnapshotType type)
{
	Snapshot snap;
	switch (type)
	{
	case F3D:
		snap.header = printF3DHeader;
		snap.body = printF3DBody;
		snap.rowDelim = printF3DRowDelim;
		snap.footer = printF3DFooter;
		break;
	case TIFFIMG:
		snap.header = printTiffHeader;
		snap.body = printTiffBody;
		snap.rowDelim = printTiffRowDelim;
		snap.footer = printTiffFooter;
		break;
	default:
	case PRINT:
		snap.header = printTerminalHeader;
		snap.body = printTerminalBody;
		snap.rowDelim = printTerminalRowDelim;
		snap.footer = printTerminalFooter;
		break;
	}
	return snap;
}

//header of the terminal snaphot - - to be passed as a function pointer (snapshotHeader)
void printTerminalHeader(Grid *g, Orientation direction, int slice)
{
	char* directionTxt = NULL;

	switch (direction)
	{
	case XY:
		directionTxt = "XY"; break;
	case XZ:
		directionTxt = "XZ"; break;
	case YZ:
		directionTxt = "YZ"; break;
	}
	printf("%s slice %d:\n", directionTxt, slice);
}

//header of the F3D snaphot - - to be passed as a function pointer (snapshotHeader)
void printF3DHeader(Grid *g, Orientation direction, int slice)
{

	sprintf(filename, "%s-x.%d.f3d", basename, frameX++);
	snapfile = fopen(filename, "w");

	fprintf(snapfile, "OPTI3DREAL\n");
	switch (direction)
	{
	case XY:
		fprintf(snapfile, "%d %d\n", g->sizeX, g->sizeY);
		fprintf(snapfile, "%d %d %d %d 0 1\n", 0, g->sizeX - 1, 0, g->sizeY - 1);
		break;
	case XZ:
		fprintf(snapfile, "%d %d\n", g->sizeX, g->sizeZ);
		fprintf(snapfile, "%d %d %d %d 0 1\n", 0, g->sizeX - 1, 0, g->sizeZ - 1);
		break;
	case YZ:
		fprintf(snapfile, "%d %d\n", g->sizeY, g->sizeZ);
		fprintf(snapfile, "%d %d %d %d 0 1\n", 0, g->sizeY - 1, 0, g->sizeZ - 1);
		break;
	}
}

//header of the TIFF snaphot - - to be passed as a function pointer (snapshotHeader)
void printTiffHeader(Grid *g, Orientation direction, int slice)
{
	
	
	sprintf(filename, "%s-x.%d.tif", basename, frameX++);
	tif = TIFFOpen(filename, "w");

	if (!tif)
	{
		perror("printTiffHeader: impossible to open ");
		exit(-1);
	}

	switch (direction)
	{
	case XY:
		width = g->sizeX; height = g->sizeY;
		break;
	case XZ:
		width = g->sizeX; height = g->sizeZ;
		break;
	case YZ:
		width = g->sizeY; height = g->sizeZ;
		break;
	}

	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);  // set the width of the image                   
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);    // set the height of the image
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);    // set the size of the channels
	TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP); //floating point tiff image
	TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
	TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

	//row buffer (scanline write tiff)
	buf = (float *)_TIFFmalloc(width * sizeof(float));
	_TIFFmemset(buf, 0, width*sizeof(float));

}

//row delimiter for terminal printout - - to be passed as a function pointer (snapshotRowDelim)
void printTerminalRowDelim(int i)
{
	printf("\n");
}

//row delimiter for terminal printout - - to be passed as a function pointer (snapshotRowDelim)
void printTiffRowDelim(int i)
{
	TIFFWriteScanline(tif, buf, i, 0);
	return;
}

//we do nothing - - to be passed as a function pointer (snapshotRowDelim)
void printF3DRowDelim(int i)
{
	return;
}

//we print the value to the terminal (tabulated) - to be passed as a function pointer (snapshotBody)
void printTerminalBody(Grid *g, float *field, int i, int j, int k, int indexer)
{
	printf(" %3.2f\t", field[idx(g, i, j, k)]);
}

//we print the values to a F3D file - to be passed as a function pointer (snapshotBody)
void printF3DBody(Grid *g, float *field, int i, int j, int k, int indexer)
{
	fprintf(snapfile, "%3.6f\n", field[idx(g, i, j, k)]);
}

//we print the values to a Tiff file - to be passed as a function pointer (snapshotBody)
void printTiffBody(Grid *g, float *field, int i, int j, int k, int indexer)
{
	buf[indexer] = field[idx(g, i, j, k)];
}

//we do nothing - to be passed as as function pointer (snapshotFooter)
void printTerminalFooter()
{
	return;
}

//we close the file at the end - to be passed as as function pointer (snapshotFooter)
void printF3DFooter()
{
	fclose(snapfile);
}

//we close the file at the end - to be passed as as function pointer (snapshotFooter)
void printTiffFooter()
{
	free(buf);
	TIFFClose(tif);
}

//generic snapshot function. To be called with function pointer for each type of snapshot
void snapshot(Grid *g, float *field, int slice, int orientation, Snapshot snap)
{

	//did we initialize the snapshots?
	if (!isInitialized()) {
		return;
	}

	//do we need to get a snapshot?
	if (!strideConditionsMet(g)) {
		return;
	}

	switch (orientation)
	{
	case XY: // XY plane
		snap.header(g, XY, slice);
		for (int i = 0; i < g->sizeX; i++)
		{
			for (int j = 0; j < g->sizeY; j++)
			{
				snap.body(g, field, i, j, slice, j);
			}
			snap.rowDelim(i);
		}
		break;
	case XZ: // XZ plane
		snap.header(g, XZ, slice);
		for (int i = 0; i < g->sizeX; i++)
		{
			for (int j = 0; j < g->sizeZ; j++)
			{
				snap.body(g, field, i, slice, j, j);
			}
			snap.rowDelim(i);
		}
		break;
	case YZ: // YZ plane
		snap.header(g, YZ, slice);
		for (int i = 0; i < g->sizeY; i++)
		{
			for (int j = 0; j < g->sizeZ; j++)
			{
				snap.body(g, field, slice, i, j, j);
			}
			snap.rowDelim(i);
		}
		break;
	}
	snap.footer();
	return;
}


