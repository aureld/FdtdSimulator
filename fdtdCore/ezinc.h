//ezinc.h: input field definitions
// Aurelien Duval 2014

#ifndef _EZINC_H_
#define _EZINC_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "fdtd-grid.h"

void ezIncInit(Grid *g);
double ezInc(double time, double location);



#endif /*_EZINC_H_*/