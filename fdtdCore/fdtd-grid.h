//fdtd-grid.h: definitions for the FDTD grid
// Aurelien Duval 2014

#ifndef _FDTD_GRID_H_
#define _FDTD_GRID_H_

enum GRIDTYPE {oneDGrid, teZGrid, tmZGrid, threeDGrid};

//FDTD grid containing all field cells and time data
struct Grid {
	double		*ex, *cexe, *cexh;		//Ex field
	double		*ey, *ceye, *ceyh;		//Ey field
	double		*ez, *ceze, *cezh;		//Ez field
	double		*hx, *chxh, *chxe;		//Hx field
	double		*hy, *chyh, *chye;		//Hy field
	double		*hz, *chzh, *chze;		//Hz field
	int			sizeX, sizeY, sizeZ;	//grid size
	int			time;		//current timestep
	int			maxTime;	//final timestep
	int			type;		// use with GRIDTYPE for 1D, 2D or 3D grids
	double		cdtds;		//courant number
};
typedef struct Grid Grid;

#endif /*_FDTD_GRID_H_*/