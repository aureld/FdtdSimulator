clear all;
close all;
clc;

c= 299792458;

Grid.nx = 100;
Grid.ny = 100;
Grid.nz = 100;
Grid.nt = 1000;

Grid.dx = 100e-9;
Grid.dy = 100e-9;
Grid.dz = 100e-9;

Grid.cfl = 1/sqrt(3);

Grid.dt = Grid.cfl * 1/(c*sqrt((1/Grid.dx).^2 + (1/Grid.dy).^2 + (1/Grid.dz).^2));



vtkwrite( 'test.vtk','structured_grid', 1:Grid.nx, 1:Grid.ny, 1:Grid.nz , 'scalars', 'material', ones(Grid.nx,1));

C