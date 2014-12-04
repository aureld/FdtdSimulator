// snapshot.c : takes snapshots at time intervals of the grid
// Aurelien Duval 2014

#include "fdtd.h"

static int temporalStride = 0;
static int spatialStride;
static int startTime;
static int startNode;
static int endNode;
static int frame = 0;
static char basename[80] = "sim";

//initializes the snapshots with hardcoded values
void snapshotInit(Grid *g)
{
	temporalStride = 10;
	startTime = 100;
	startNode = 0;
	endNode = 250;
	spatialStride = 10;
	return;
}

//captures a snapshot of the grid if conditions are met
void snapshot(Grid *g)
{
	int mm;
	char filename[100];
	FILE *snapshot;

	if (temporalStride <= 0) {
		fprintf(stderr, "snapshot: uninitialized snapshots, temporal stride should be > 0.\n");
		exit(-1);
	}

	if (Time >= startTime && (Time - startTime) % temporalStride == 0)
	{
		sprintf(filename, "%s.%d", basename, frame++);
		snapshot = fopen(filename, "w");

		for (mm = 0; mm <= endNode; mm += spatialStride)
			fprintf(snapshot, "%g\n", Ez(mm));
		fclose(snapshot);
	}

	return;
}



