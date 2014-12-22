//fdtd-protos.h: functions prototypes
// Aurelien Duval 2014

#pragma once
#ifndef _FDTD_PROTOS_H_
#define _FDTD_PROTOS_H_

#include "fdtd-macros.h"

//prototypes
void gridInit(Grid *g);
void updateH(Grid *g);
void updateE(Grid *g);
void abcInit(Grid *g);
void abc(Grid *g);

void snapshotInit(Grid *g);
void snaphotInitStartTime(int start);
void snaphotInitTemporalStride(int stride);
Snapshot snapshotSetType(SnapshotType type);
void snapshot(Grid *g, float *field, int slice, int orientation, Snapshot snap);
void printTerminalHeader(Grid *g, Orientation direction, int slice);
void printF3DHeader(Grid *g, Orientation direction, int slice);
void printTiffHeader(Grid *g, Orientation direction, int slice);
void printTerminalRowDelim(int i);
void printF3DRowDelim(int i);
void printTiffRowDelim(int i);
void printTerminalBody(Grid *g, float *field, int i, int j, int k, int indexer);
void printF3DBody(Grid *g, float *field, int i, int j, int k, int indexer);
void printTiffBody(Grid *g, float *field, int i, int j, int k, int indexer);
void printF3DFooter();
void printTiffFooter();
void printTerminalFooter();


void updateEx(Grid *g);
void updateEy(Grid *g);
void updateEz(Grid *g);
void updateHx(Grid *g);
void updateHy(Grid *g);
void updateHz(Grid *g);
void ezIncInit(Grid *g);
float ezInc(int time, int location);

void do_time_stepping(Grid *g, Snapshot *snapshots);
float *field(Grid *g, FieldType f);


#endif /*_FDTD_PROTOS_H_*/
