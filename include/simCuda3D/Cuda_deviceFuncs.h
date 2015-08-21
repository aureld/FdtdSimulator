//Cuda_deviceFuncs.h: functions executed on device side only
// Aurelien Duval 2015

#include <cuda_runtime.h>


__device__ inline void Cuda_updateHComponent(int component, float *h, int i, int j, int k, int pos, float *ex, float *ey, float *ez, float *Db1, float *Db2);

