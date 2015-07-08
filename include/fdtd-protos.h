//fdtd-protos.h: functions prototypes
// Aurelien Duval 2014

#pragma once
#ifndef _FDTD_PROTOS_H_
#define _FDTD_PROTOS_H_


#include "fdtd-macros.h"


//forward declarations
class Movie;

//prototypes
void gridInit(Grid *g, int, int, int, int);
void updateH(Grid *g);
void updateE(Grid *g);
void abcInit(Grid *g);
void abc(Grid *g);

void updateHx(Grid *g);
void updateHy(Grid *g);
void updateHz(Grid *g);

void updateEx(Grid *g);
void updateEy(Grid *g);
void updateEz(Grid *g);

void ezIncInit(Grid *g);
float ezInc(int time, double delay);

void do_time_stepping(Grid *g,  Movie *movie);
float *field(Grid *g, FieldType f);


#endif /*_FDTD_PROTOS_H_*/
