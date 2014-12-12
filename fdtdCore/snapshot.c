// snapshot.c : takes snapshots at time intervals of the grid
// Aurelien Duval 2014

#include <stdio.h>
#include <stdlib.h>
#include "fdtd-macros.h"

static int temporalStride = -2;
static int startTime;
static int frameX = 0, frameY = 0;
static char basename[80] = "results\\sim";

//initializes the snapshots with hardcoded values
void snapshotInit(Grid *g)
{
	temporalStride = 1;
	startTime = 0;
	return;
}

//captures a snapshot of the grid if conditions are met
void snapshotF3D(Grid *g)
{
	int mm, nn, pp;
	float temp;
	int dim1, dim2;
	char filename[100];
	FILE *snapshot;

	if (temporalStride <= 0) {
		fprintf(stderr, "snapshot: uninitialized snapshots, temporal stride should be > 0.\n");
		exit(-1);
	}

	if (g->time >= startTime && (g->time - startTime) % temporalStride == 0)
	{
		/* Write the X = center slice */
		sprintf(filename, "%s-x.%d.f3d", basename, frameX++);
		snapshot = fopen(filename, "w");

		//we store everything as floats for output

		//dimensions first
		dim1 = g->sizeY;
		dim2 = g->sizeZ;
		fprintf(snapshot, "OPTI3DREAL\n");
		fprintf(snapshot, "%d %d\n", dim1, dim2);
		fprintf(snapshot, "%d %d %d %d 0 1\n", 0, dim1 - 1, 0, dim2 - 1);
		//write the data itself
		mm = (g->sizeX) / 2;
		for (pp = g->sizeZ - 1; pp >= 0; pp--)
			for (nn = 0; nn < dim1; nn++)
			{
				temp = g->ex[idx(g,mm, nn, pp)];
				fprintf(snapshot, "%f\n", temp);
			}

		fclose(snapshot);
		
		
		
		///* Write the Y = center slice */
		//sprintf(filename, "%s-y.%d.f3d", basename, frameY++);
		//snapshot = fopen(filename, "w");

		////we store everything as floats for output

		////dimensions first
		//dim1 = SizeX - 1;
		//dim2 = SizeZ;
		//fprintf(snapshot, "OPTI3DREAL\n");
		//fprintf(snapshot, "%d %d\n", dim1, dim2);
		//fprintf(snapshot, "%d %d %d %d 0 1\n", 0, SizeX - 2, 0, SizeZ - 1);
		////write the data itself
		//nn = SizeY / 2;
		//for (pp = SizeZ - 1; pp >= 0; pp--)
		//	for (mm = 0; mm < SizeX; mm++)
		//	{
		//	temp = (float)Ex(mm, nn, pp);
		//	fprintf(snapshot, "%f\n", temp);
		//	}
		//
		//fclose(snapshot);
	}
	
	return;
}



void print(Grid *g, int slice, int orientation)
{
	switch (orientation)
	{
	case 0: // XY plane
		printf("XY slice %d, Ex grid:\n", slice);
		for (int i = 0; i < g->sizeX; i++)
		{
			for (int j = 0; j < g->sizeY; j++)
			{
				printf(" %3.2g\t", g->ex[idx(g,i, j, slice)]);
			}
			printf("\n");
		}
		break;
	/*case 1: //XZ plane
		System.Console.WriteLine("XZ slice # {0}, Ex grid:", slice);
		for (int i = 0; i < Ex.GetLength(0); i++)
		{
			for (int k = 0; k < Ex.GetLength(2); k++)
				System.Console.Write(" {0:E}\t", Ex[i, slice, k]);
			System.Console.WriteLine();
		}
		break;
	case 2: //YZ plane
		System.Console.WriteLine("YZ slice # {0}, Ex grid:", slice);
		for (int j = 0; j < Ex.GetLength(1); j++)
		{
			for (int k = 0; k < Ex.GetLength(2); k++)
				System.Console.Write(" {0:E}\t", Ex[slice, j, k]);
			System.Console.WriteLine();
		}
		break;*/
	}

}



