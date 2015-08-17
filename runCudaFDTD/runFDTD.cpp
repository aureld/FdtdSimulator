// runFDTD.cu : FDTD core engine in C, CUDA variant
// see http://www.eecs.wsu.edu/~schneidj
// Aurelien Duval 2015

#include <stdio.h>
#include <iostream>
#include "SimCuda3D/Cuda_grid.h"
#include "SimCuda3D/Cuda_macros.h"
#include "SimCuda3D/SimCudaFunctions.h"
#include "JsonParser/JsonParser.h"
#include "JsonParser/JsonDocument.h"
#include "JsonParser/FileIO.h"

using namespace std;



int main() 
{
    grid *g;
    g = (grid*) calloc(1, sizeof(grid));


    JsonDocument *doc = new JsonDocument();
    FileIO *fio = new FileIO();
    JsonParser *parser = new JsonParser(doc, fio);
    grid *g = new grid();
    parser->ParseJsonFile("ValidTestJson.txt", g);
    return 0;
}