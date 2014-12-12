//fdtd-protos.h: functions prototypes
// Aurelien Duval 2014

#ifndef _FDTD_PROTOS_H_
#define _FDTD_PROTOS_H_

#include "fdtd-grid.h"

//prototypes
void gridInit(Grid *g);
void updateH(Grid *g);
void updateE(Grid *g);
void abcInit(Grid *g);
void abc(Grid *g);
void snapshotInit(Grid *g);
void snapshotF3D(Grid *g);
void print(Grid *g, int slice, int orientation);
void updateEx(Grid *g);
void updateEy(Grid *g);
void updateEz(Grid *g);
void updateHx(Grid *g);
void updateHy(Grid *g);
void updateHz(Grid *g);
void ezIncInit(Grid *g);
float ezInc(int time, int location);


#endif /*_FDTD_PROTOS_H_*/