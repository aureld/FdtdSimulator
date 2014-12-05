// snapshot.c : takes snapshots at time intervals of the grid
// Aurelien Duval 2014

#include <stdio.h>
#include <stdlib.h>
#include "fdtd-macros.h"

static int temporalStride = 0;
static int startTime;
static int startNodeX, endNodeX, spatialStrideX;
static int startNodeY, endNodeY, spatialStrideY;
static int frame = 0;
static char basename[80] = "results\\sim";

//initializes the snapshots with hardcoded values
void snapshotInit(Grid *g)
{
	temporalStride = 1;
	startTime = 0;
	startNodeX = 0;
	endNodeX = 100;
	spatialStrideX = 1;
	startNodeY = 0;
	endNodeY = 80;
	spatialStrideY = 1;
	return;
}

//captures a snapshot of the grid if conditions are met
void snapshot(Grid *g)
{
	int mm, nn;
	float temp;
	int dim1, dim2;
	char filename[100];
	FILE *snapshot;

	if (temporalStride <= 0) {
		fprintf(stderr, "snapshot: uninitialized snapshots, temporal stride should be > 0.\n");
		exit(-1);
	}

	if (Time >= startTime && (Time - startTime) % temporalStride == 0)
	{
		sprintf(filename, "%s.%d.f3d", basename, frame++);
		snapshot = fopen(filename, "w");

		//we store everything as floats for output

		//dimensions first
		dim1 = (endNodeX - startNodeX) / spatialStrideX + 1;
		dim2 = (endNodeY - startNodeY) / spatialStrideY + 1;
		//fwrite(&dim1, sizeof(float), 1, snapshot);
		//fwrite(&dim2, sizeof(float), 1, snapshot);
		fprintf(snapshot, "OPTI3DREAL\n");
		fprintf(snapshot, "%d %d\n", dim1, dim2);
		fprintf(snapshot, "%d %d %d %d 0 1\n", startNodeX, endNodeX, startNodeY, endNodeY);


		//write the data itself
		for (nn = endNodeY; nn >= startNodeY; nn -= spatialStrideY)
			for (mm = startNodeX; mm <= endNodeX; mm += spatialStrideX)
			{
			temp = (float)Ez(mm, nn);
			//fwrite(&temp, sizeof(float), 1, snapshot);
			fprintf(snapshot, "%f\n", temp);
			}
		fclose(snapshot);
	}

	return;
}



