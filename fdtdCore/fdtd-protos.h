//fdtd-protos.h: functions prototypes
// Aurelien Duval 2014

#ifndef _FDTD_PROTOS_H_
#define _FDTD_PROTOS_H_

#include "fdtd-grid.h"

//prototypes
void gridInit(Grid *g);
void updateH(Grid *g);
void updateE(Grid *g);
/*
void abcInit(Grid *g);
void abc(Grid *g);

void tfsfInit(Grid *g);
void tfsfUpdate(Grid *g);
*/
void snapshotInit(Grid *g);
void snapshot(Grid *g);

#endif /*_FDTD_PROTOS_H_*/