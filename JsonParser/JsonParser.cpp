//JsonParser.cpp: Functions used for parsing JSON files containing FDTD simulation parameters
// Aurelien Duval 2015

#include <stdio.h>
#include "JsonParser/JSonParser.h"
#include "rapidJson/filereadstream.h"
#include "rapidJson/filewritestream.h"
#include "rapidJson/prettywriter.h"
#include <cstdio>
#include <iostream>
#include <fstream>
#include "simCuda3D/Cuda_grid.h"

using namespace rapidjson;
using namespace std;


//Writes the content of the grid to a Json file given by filename
bool JsonParser::ExportToJson(const char* filename, grid *g)
{
    FILE *myfile = fio->Open(filename, "w");
    if (myfile == NULL)
    {
        perror("[JsonParser]: Error creating file");
        return false;
    }

    if (g == NULL)
    {
        perror("[JsonParser]: please provide an initialized grid");
        return false;
    }

    if (doc->WriteToDocument(myfile, g) == false)
    {
        perror("[JsonParser]: invalid grid");
        return false;
    }

    fio->Close(myfile);
    return true;
}

//parse Json file containing the parameters for the simulation. Fills in the grid structure
bool JsonParser::ParseJsonFile(const char* filename, grid *g)
{
    FILE *myfile = fio->Open(filename, "r");
    if (myfile == NULL)
    {   
        perror("[JsonParser]: Error opening file");
        return false;
    }

    if (g == NULL)
    {
        perror("[JsonParser]: please provide an initialized grid");
        return false;
    }
    
    char readBuffer[65536];
    FileReadStream is(myfile, readBuffer, sizeof(readBuffer));

    doc->ParseStream(is); //parse the file and fill the DOM
    if (doc->HasParseError())
    {
        perror("[JsonParser]: invalid Json document");
        return false;
    }

    if (doc->WriteToGrid(g) == false) //fill the grid struct with DOM elements
    {
        perror("[JsonParser]: invalid Json schema");
        return false;
    }
    fio->Close(myfile);
    return true;
}



