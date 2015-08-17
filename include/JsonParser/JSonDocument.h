//JSonDocument.h: Json Document interface to RapidJson
//Aurelien Duval 2015

#pragma once
#ifndef _JSONDOCUMENT_H_
#define _JSONDOCUMENT_H_

#include "RapidJson/document.h"
#include "RapidJson/filewritestream.h"
#include "RapidJson/filereadstream.h"
#include "SimCuda3D/cuda_grid.h"

using namespace rapidjson;

class IJsonDocument {
public:
    virtual ~IJsonDocument() {}
    virtual Document& ParseStream(FileReadStream is) = 0; //from RapidJson
    virtual bool HasParseError() = 0; //from RapidJson

    virtual bool WriteToGrid(grid *g) = 0;
    virtual bool WriteToDocument(FILE *f, grid *g) = 0;

};


class JsonDocument : public IJsonDocument, public Document {
public:
    virtual Document& ParseStream(FileReadStream is) { return doc.ParseStream(is); };
    virtual bool HasParseError() { return doc.HasParseError(); };

    virtual bool WriteToGrid(grid *g);
    virtual bool WriteToDocument(FILE *f,grid *g);


protected:
    Document doc;
};



#endif //_JSONDOCUMENT_H_