//JsonParser.h: Functions used for parsing JSON files containing FDTD simulation parameters
// Aurelien Duval 2015

#pragma once
#ifndef _JSONPARSER_H_
#define _JSONPARSER_H_

#include "simCuda3D/Cuda_grid.h"
#include "JsonParser/FileIO.h"
#include "JsonParser/JsonDocument.h"


class JsonParser
{
public:
    JsonParser(IJsonDocument *document, IFileIO *f) { doc = document; fio = f; }
    ~JsonParser() {};
    bool ParseJsonFile(const char* filepath, grid *g);
    bool ExportToJson(const char* filename, grid *g);
    
private:
    static bool WriteJsonDocToGrid(Document d, grid *g);
    IFileIO *fio;
    IJsonDocument *doc;
};





#endif /*_JSONPARSER_H_*/