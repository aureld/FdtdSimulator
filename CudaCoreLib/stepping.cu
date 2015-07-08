
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Cuda_fdtd-protos.h"
#include "Cuda_fdtd_alloc.h"
#include "Cuda_fdtd_macros.h"
#include "fdtd-macros.h"
#include "MovieLib.h"

#include <stdio.h>

//"private" functions
void PrepareFrame(Grid *g, unsigned char * buf);
double interpolate(double val, double y0, double x0, double y1, double x1);
double base(double val);
double red(double gray);
double green(double gray);
double blue(double gray);


void Cuda_do_time_stepping(Grid *g, Movie *movie)
{

    unsigned char *buf = new unsigned char[g->sizeX * g->sizeY * 3];

    //setup cuda grid and dimensions
	// cudaDeviceProp prop;
    //HANDLE_ERROR( cudaGetDeviceProperties(&prop, 0));
	//int threadsToLaunch = g->sizeX * g->sizeY * g->sizeZ;

	
	int nx = g->sizeX;
	int ny = g->sizeY;
	int nz = g->sizeZ;
	int linsize = nx*ny*nz;

	dim3 blockSize(nz, 1);
	dim3 gridSize(nx,ny);


	float		*ex, *cexe, *cexh;		//Ex field
	float		*ey, *ceye, *ceyh;		//Ey field
	float		*ez, *ceze, *cezh;		//Ez field
	float		*hx, *chxh, *chxe;		//Hx field
	float		*hy, *chyh, *chye;		//Hy field
	float		*hz, *chzh, *chze;		//Hz field

	//memory allocation
	CUDA_ALLOC_3D(hx, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(chxh, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(chxe, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(hy, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(chyh, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(chye, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(hz, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(chzh, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(chze, g->sizeX, g->sizeY, g->sizeZ, float);

	CUDA_ALLOC_3D(ex, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(cexe, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(cexh, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(ey, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(ceye, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(ceyh, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(ez, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(ceze, g->sizeX, g->sizeY, g->sizeZ, float);
	CUDA_ALLOC_3D(cezh, g->sizeX, g->sizeY, g->sizeZ, float);

	HANDLE_ERROR(cudaMemcpy(ex	, g->ex		, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(cexe, g->cexe	, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(cexh, g->cexh	, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ey	, g->ey		, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ceye, g->ceye	, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ceyh, g->ceyh	, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ez	, g->ez		, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ceze, g->ceze	, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(cezh, g->cezh	, linsize*sizeof(float), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(hx	, g->hx		, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(chxe, g->chxe	, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(chxh, g->chxh	, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(hy	, g->hy		, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(chye, g->chye	, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(chyh, g->chyh	, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(hz	, g->hz		, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(chze, g->chze	, linsize*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(chzh, g->chzh	, linsize*sizeof(float), cudaMemcpyHostToDevice));



    for (g->time = 0; g->time < g->maxTime; g->time++)
    {
        //update fields
		Cuda_updateHx <<< gridSize, blockSize >>>		(ex, cexe, cexh, ey, ceye, ceyh, ez, ceze, cezh,
														hx, chxe, chxh, hy, chye, chyh, hz, chze, chzh,
														nx, ny, nz, g->time);
		Cuda_updateHy << < gridSize, blockSize >> >		(ex, cexe, cexh, ey, ceye, ceyh, ez, ceze, cezh,
														hx, chxe, chxh, hy, chye, chyh, hz, chze, chzh,
														nx, ny, nz, g->time);
		Cuda_updateHz << < gridSize, blockSize >> >		(ex, cexe, cexh, ey, ceye, ceyh, ez, ceze, cezh,
														hx, chxe, chxh, hy, chye, chyh, hz, chze, chzh,
														nx, ny, nz, g->time);

		HANDLE_ERROR(cudaDeviceSynchronize());

		Cuda_updateEx << < gridSize, blockSize >> >		(ex, cexe, cexh, ey, ceye, ceyh, ez, ceze, cezh,
														hx, chxe, chxh, hy, chye, chyh, hz, chze, chzh,
														nx, ny, nz, g->time);
		Cuda_updateEy << < gridSize, blockSize >> >		(ex, cexe, cexh, ey, ceye, ceyh, ez, ceze, cezh,
														hx, chxe, chxh, hy, chye, chyh, hz, chze, chzh,
														nx, ny, nz, g->time);
		Cuda_updateEz << < gridSize, blockSize >> >		(ex, cexe, cexh, ey, ceye, ceyh, ez, ceze, cezh,
														hx, chxe, chxh, hy, chye, chyh, hz, chze, chzh,
														nx, ny, nz, g->time);
		HANDLE_ERROR(cudaDeviceSynchronize());

		

		// source
		int posX = g->sizeX / 2;
		int posY = g->sizeY / 2; 
		int posZ = g->sizeZ / 2;
		double delay = 1.0;

		Cuda_applySource <<< 1, 1 >>> (ex, ey, ez, posX, posY, posZ, nx, ny, nz, delay, g->time);
		HANDLE_ERROR(cudaDeviceSynchronize());

		if (movie)
		{
			//we copy the field we want to write to host mem
			HANDLE_ERROR(cudaMemcpy(g->ey, ey, linsize*sizeof(float), cudaMemcpyDeviceToHost));
			PrepareFrame(g, buf);
			movie->SetData(buf);
			movie->Write();
		}

        //update console display
        printf("time: %d / %d\n", g->time, g->maxTime - 1);
    }
 
	HANDLE_ERROR(cudaMemcpy(g->ex,		ex, linsize*sizeof(float),	 cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->cexe,	cexe, linsize*sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->cexh,	cexh, linsize*sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->ey,		ey, linsize*sizeof(float),	 cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->ceye,	ceye, linsize*sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->ceyh,	ceyh, linsize*sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->ez,		ez, linsize*sizeof(float),	 cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->ceze,	ceze, linsize*sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->cezh,	cezh, linsize*sizeof(float), cudaMemcpyDeviceToHost));
																	 
	HANDLE_ERROR(cudaMemcpy(g->hx,		hx, linsize*sizeof(float),	 cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->chxe,	chxe, linsize*sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->chxh,	chxh, linsize*sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->hy,		hy, linsize*sizeof(float),	 cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->chye,	chye, linsize*sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->chyh,	chyh, linsize*sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->hz,		hz, linsize*sizeof(float),	 cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->chze,	chze, linsize*sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(g->chzh,	chzh, linsize*sizeof(float), cudaMemcpyDeviceToHost));

	delete[] buf;
	HANDLE_ERROR(cudaFree(ex));	  HANDLE_ERROR(cudaFree(ey)); HANDLE_ERROR(cudaFree(ez));
	HANDLE_ERROR(cudaFree(cexe)); HANDLE_ERROR(cudaFree(cexh));
	HANDLE_ERROR(cudaFree(ceye)); HANDLE_ERROR(cudaFree(ceyh));
	HANDLE_ERROR(cudaFree(ceze)); HANDLE_ERROR(cudaFree(cezh));
	HANDLE_ERROR(cudaFree(hx));   HANDLE_ERROR(cudaFree(hy)); HANDLE_ERROR(cudaFree(hz));
	HANDLE_ERROR(cudaFree(chxe)); HANDLE_ERROR(cudaFree(chxh)); 
	HANDLE_ERROR(cudaFree(chye)); HANDLE_ERROR(cudaFree(chyh));
	HANDLE_ERROR(cudaFree(chze)); HANDLE_ERROR(cudaFree(chzh));

	return;
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
            double val = (g->ey[i] - mincolors) * normfact;
            buf[pos] = red(val); //R
            buf[pos + 1] = green(val); //G
            buf[pos + 2] = blue(val); //B
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
    return base(gray - 0.5) * 255;
}
double green(double gray) {
    return base(gray) * 255;
}
double blue(double gray) {
    return base(gray + 0.5) * 255;
}