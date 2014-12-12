//fdtd-macros.h: Macros used for accessing arrays and structs
// Aurelien Duval 2014

#ifndef _FDTD_MACROS_H_
#define _FDTD_MACROS_H_

#include "fdtd-grid.h"

//Constants
# define M_PI 3.14159265358979323846 /* pi */



//Array macros

/* Explicit 1D Grid - G is the grid */
#define Hy1G(G, M)		G->hy[M]
#define Chyh1G(G, M)	G->chyh[M]
#define Chye1G(G, M)	G->chye[M]

#define Ez1G(G, M)		G->ez[M]
#define Ceze1G(G, M)	G->ceze[M]
#define Cezh1G(G, M)	G->cezh[M]

/* Explicit  2D TMz Grid - G is the grid */
#define Hx2G(G, M, N)	G->hx[  (M) * (SizeYG(G)-1) + (N)]
#define Chxh2G(G, M, N)	G->chxh[(M) * (SizeYG(G)-1) + (N)]
#define Chxe2G(G, M, N)	G->chxe[(M) * (SizeYG(G)-1) + (N)]

#define Hy2G(G, M, N)	G->hy[(M) *	  SizeYG(G) + (N)]
#define Chyh2G(G, M, N)	G->chyh[(M) * SizeYG(G) + (N)]
#define Chye2G(G, M, N)	G->chye[(M) * SizeYG(G) + (N)]

#define Ez2G(G, M, N)	G->ez[(M) *   SizeYG(G) + (N)]
#define Ceze2G(G, M, N)	G->ceze[(M) * SizeYG(G) + (N)]
#define Cezh2G(G, M, N)	G->cezh[(M) * SizeYG(G) + (N)]


/* Explicit  2D TEz Grid - G is the grid */
#define Ex2G(G, M, N)	G->ex[  (M) * (SizeYG(G)) + (N)]
#define Cexe2G(G, M, N)	G->cexe[(M) * (SizeYG(G)) + (N)]
#define Cexh2G(G, M, N)	G->cexh[(M) * (SizeYG(G)) + (N)]

#define Ey2G(G, M, N)	G->ey[(M)	* (SizeYG(G)-1) + (N)]
#define Ceye2G(G, M, N)	G->ceye[(M) * (SizeYG(G)-1) + (N)]
#define Ceyh2G(G, M, N)	G->ceyh[(M) * (SizeYG(G)-1) + (N)]

#define Hz2G(G, M, N)	G->hz[(M)	* (SizeYG(G)-1) + (N)]
#define Chzh2G(G, M, N)	G->chzh[(M) * (SizeYG(G)-1) + (N)]
#define Chze2G(G, M, N)	G->chze[(M) * (SizeYG(G)-1) + (N)]


/* Explicit 3D grid - G is the grid */
//#define HxG(G, M, N, P)		G->hx[	((M) * (SizeYG(G)-1) + (N)) * (SizeZG(G)-1) + P]

#define ChxhG(G, M, N, P)	G->chxh[((M) * (SizeYG(G)-1) + (N)) * (SizeZG(G)-1) + P]
#define ChxeG(G, M, N, P)	G->chxe[((M) * (SizeYG(G)-1) + (N)) * (SizeZG(G)-1) + P]

#define HyG(G, M, N, P)		G->hy[	((M) * (SizeYG(G)) + (N))	* (SizeZG(G)-1) + P]
#define ChyhG(G, M, N, P)	G->chyh[((M) * (SizeYG(G)) + (N))	* (SizeZG(G)-1) + P]
#define ChyeG(G, M, N, P)	G->chye[((M) * (SizeYG(G)) + (N))	* (SizeZG(G)-1) + P]

#define HzG(G, M, N, P)		G->hz[	((M) * (SizeYG(G)-1) + (N)) * (SizeZG(G)) + P]
#define ChzhG(G, M, N, P)	G->chzh[((M) * (SizeYG(G)-1) + (N)) * (SizeZG(G)) + P]
#define ChzeG(G, M, N, P)	G->chze[((M) * (SizeYG(G)-1) + (N)) * (SizeZG(G)) + P]

#define ExG(G, M, N, P)		G->ex[	((M) * (SizeYG(G)) + (N))	* (SizeZG(G)) + P]
#define CexeG(G, M, N, P)	G->cexe[((M) * (SizeYG(G)) + (N))	* (SizeZG(G)) + P]
#define CexhG(G, M, N, P)	G->cexh[((M) * (SizeYG(G)) + (N))	* (SizeZG(G)) + P]

#define EyG(G, M, N, P)		G->ey[	((M) * (SizeYG(G)-1) + (N))	* (SizeZG(G)) + P]
#define CeyeG(G, M, N, P)	G->ceye[((M) * (SizeYG(G)-1) + (N))	* (SizeZG(G)) + P]
#define CeyhG(G, M, N, P)	G->ceyh[((M) * (SizeYG(G)-1) + (N))	* (SizeZG(G)) + P]

#define EzG(G, M, N, P)		G->ez[	((M) * (SizeYG(G)) + (N))	* (SizeZG(G)-1) + P]
#define CezeG(G, M, N, P)	G->ceze[((M) * (SizeYG(G)) + (N))	* (SizeZG(G)-1) + P]
#define CezhG(G, M, N, P)	G->cezh[((M) * (SizeYG(G)) + (N))	* (SizeZG(G)-1) + P]

/* Explicit general simulation parameters */
#define SizeXG(G)		G->sizeX
#define SizeYG(G)		G->sizeY
#define SizeZG(G)		G->sizeZ
#define TimeG(G)		G->time
#define MaxTimeG(G)		G->maxTime
#define CdtdsG(G)		G->cdtds
#define TypeG(G)		G->type


/* Implicit 1D Grid - g is the grid */
#define Hy1(M)				Hy1G(g,M)
#define Chyh1(M)			Chyh1G(g,M)
#define Chye1(M)			Chye1G(g,M)

#define Ez1(M)				Ez1G(g,M)
#define Ceze1(M)			Ceze1G(g,M)
#define Cezh1(M)			Cezh1G(g,M)

/* Implicit 2D TMz Grid - g is the grid */
#define Hx2(M,N)			Hx2G(g,M,N)
#define Chxh2(M,N)			Chxh2G(g,M,N)
#define Chxe2(M,N)			Chxe2G(g,M,N)

#define Hy2(M,N)			Hy2G(g,M,N)
#define Chyh2(M,N)			Chyh2G(g,M,N)
#define Chye2(M,N)			Chye2G(g,M,N)

#define Ez2(M,N)			Ez2G(g,M,N)
#define Ceze2(M,N)			Ceze2G(g,M,N)
#define Cezh2(M,N)			Cezh2G(g,M,N)

/* Implicit 2D TEz Grid - g is the grid */
#define Ex2(M,N)			Ex2G(g,M,N)
#define Cexe2(M,N)			Cexe2G(g,M,N)
#define Cexh2(M,N)			Cexh2G(g,M,N)

#define Ey2(M,N)			Ey2G(g,M,N)
#define Ceye2(M,N)			Ceye2G(g,M,N)
#define Ceyh2(M,N)			Ceyh2G(g,M,N)

#define Hz2(M,N)			Hz2G(g,M,N)
#define Chzh2(M,N)			Chzh2G(g,M,N)
#define Chze2(M,N)			Chze2G(g,M,N)

/* Implicit 3D grid - g is the grid */
#define Hx(M,N,P)			HxG(g,M,N,P)
#define Chxh(M,N,P)			ChxhG(g,M,N,P)
#define Chxe(M,N,P)			ChxeG(g,M,N,P)

#define Hy(M,N,P)			HyG(g,M,N,P)
#define Chyh(M,N,P)			ChyhG(g,M,N,P)
#define Chye(M,N,P)			ChyeG(g,M,N,P)

#define Hz(M,N,P)			HzG(g,M,N,P)
#define Chzh(M,N,P)			ChzhG(g,M,N,P)
#define Chze(M,N,P)			ChzeG(g,M,N,P)

#define Ex(M,N,P)			ExG(g,M,N,P)
#define Cexe(M,N,P)			CexeG(g,M,N,P)
#define Cexh(M,N,P)			CexhG(g,M,N,P)

#define Ey(M,N,P)			EyG(g,M,N,P)
#define Ceye(M,N,P)			CeyeG(g,M,N,P)
#define Ceyh(M,N,P)			CeyhG(g,M,N,P)

#define Ez(M,N,P)			EzG(g,M,N,P)
#define Ceze(M,N,P)			CezeG(g,M,N,P)
#define Cezh(M,N,P)			CezhG(g,M,N,P)

/* implicit general simulation parameters  - g is the grid */
#define SizeX		SizeXG(g)
#define SizeY		SizeYG(g)
#define SizeZ		SizeZG(g)
#define Time		TimeG(g)
#define	MaxTime		MaxTimeG(g)
#define Cdtds		CdtdsG(g)
#define Type		TypeG(g)


__inline long idx(Grid *g, int i, int j, int k)
{
	return (long)i * g->sizeY * g->sizeZ + (long)j * g->sizeZ + (long)k;
}



#endif /*_FDTD_MACROS_H_*/