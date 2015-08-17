// stepping.c : performs FDTD time-stepping
// Aurelien Duval 2014

#include <stdio.h>
#include "fdtd-protos.h"
#include "fdtd-macros.h"
#include "MovieLib.h"
#include <math.h>


//"private" functions
void PrepareFrame(Grid *g, unsigned char * buf);
double interpolate(double val, double y0, double x0, double y1, double x1);
double base(double val);
double red(double gray);
double green(double gray);
double blue(double gray);




//Main time stepping loop
void do_time_stepping(Grid *g, Movie *movie)
{
	unsigned char *buf = new unsigned char[g->sizeX * g->sizeY * 3];
	for (g->time = 0; g->time < g->maxTime; g->time++)
	{
		//update fields
		updateH(g);
		updateE(g);

		// source
		g->ex[idx(g, g->sizeX /2, g->sizeY / 2, g->sizeZ / 2)] += ezInc(g->time, 1.0);

		//writes movie frame for the current step
		if (movie)
		{
			PrepareFrame(g, buf);
			movie->SetData(buf);
			movie->Write();
		}

		//update console display
		printf("time: %d / %d\n", g->time, g->maxTime - 1);
	}
	delete[] buf;
}

float * field(Grid *g, FieldType f)
{
	switch (f)
	{
    case EX: default: return g->ex; break;
	case EY: return g->ey; break;
	case EZ: return g->ez; break;
	case HX: return g->hx; break;
	case HY: return g->hy; break;
	case HZ: return g->hz; break;
	}
}

//fills the frame with values from the fields
void PrepareFrame(Grid *g, unsigned char * buf)
{
	int pos = 0;
	int i;
	double maxcolors = 1;
	double mincolors = 0.0;
	double normfact = 255.0 / (maxcolors - mincolors);
	for (int y = 0; y < g->sizeY; y++)
		for (int x = 0; x <g->sizeX; x++)
		{
			pos = 3 * (y * g->sizeX + x);
			i = idx(g, x, y, g->sizeZ / 2);
			double val = (g->ex[i] - mincolors) * normfact;
			buf[pos] =      (unsigned char) red(val); //R
            buf[pos + 1] =  (unsigned char)green(val); //G
            buf[pos + 2] =  (unsigned char)blue(val); //B
		}
}


double interpolate(double val, double y0, double x0, double y1, double x1) {
	return (val - x0)*(y1 - y0) / (x1 - x0) + y0;
}

double base(double val) {
	if (val <= -0.75) return 0;
	else if (val <= -0.25) return interpolate(val, 0.0, -0.75, 1.0, -0.25);
	else if (val <= 0.25) return 1.0;
	else if (val <= 0.75) return interpolate(val, 1.0, 0.25, 0.0, 0.75);
	else return 0.0;
}

double red(double gray) {
	return base(gray - 0.5)*255;
}
double green(double gray) {
	return base(gray)*255;
}
double blue(double gray) {
	return base(gray + 0.5)*255;
}