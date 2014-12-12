//fdtd-grid.h: definitions for the FDTD grid
// Aurelien Duval 2014

#ifndef _FDTD_GRID_H_
#define _FDTD_GRID_H_

enum GRIDTYPE {oneDGrid, teZGrid, tmZGrid, threeDGrid};

//FDTD grid containing all field cells and time data
struct Grid {
	float		*ex, *cexe, *cexh;		//Ex field
	float		*ey, *ceye, *ceyh;		//Ey field
	float		*ez, *ceze, *cezh;		//Ez field
	float		*hx, *chxh, *chxe;		//Hx field
	float		*hy, *chyh, *chye;		//Hy field
	float		*hz, *chzh, *chze;		//Hz field
	int			sizeX, sizeY, sizeZ;	//grid size
	int			time;		//current timestep
	int			maxTime;	//final timestep
	int			type;		// use with GRIDTYPE for 1D, 2D or 3D grids
	float		cdtds;		//courant number
};
typedef struct Grid Grid;

#endif /*_FDTD_GRID_H_*/