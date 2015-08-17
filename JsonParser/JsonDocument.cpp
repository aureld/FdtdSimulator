//JsonDocument.cpp: Json Document interface to RapidJson
//Aurelien Duval 2015

#include "JsonParser/JsonDocument.h"
#include "rapidJson/filewritestream.h"
#include "rapidJson/prettywriter.h"


//maps document names/values to fields in the grid structure
//TODO: find a better solution for parsing the doc.
//the return type of FindMember makes it quite complex to test properly
bool JsonDocument::WriteToGrid(grid *g)
{
    Value::ConstMemberIterator itr;

    itr = doc.FindMember("nx"); g->nx = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
    itr = doc.FindMember("ny"); g->ny = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
    itr = doc.FindMember("nz"); g->nz = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
    itr = doc.FindMember("domainSize"); g->domainSize = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
    itr = doc.FindMember("layoutx"); g->layoutx = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
    itr = doc.FindMember("layouty"); g->layouty = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
    itr = doc.FindMember("layoutz"); g->layoutz = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
    itr = doc.FindMember("offset"); g->offset = (itr != doc.MemberEnd()) ? itr->value.GetInt() : NULL;
    itr = doc.FindMember("nt"); g->nt = (itr != doc.MemberEnd()) ? itr->value.GetUint64() : NULL;
    itr = doc.FindMember("dt"); g->dt = (itr != doc.MemberEnd()) ? itr->value.GetDouble() : NULL;
    itr = doc.FindMember("dx"); g->dx = (itr != doc.MemberEnd()) ? itr->value.GetDouble() : NULL;
    itr = doc.FindMember("dy"); g->dy = (itr != doc.MemberEnd()) ? itr->value.GetDouble() : NULL;
    itr = doc.FindMember("dz"); g->dz = (itr != doc.MemberEnd()) ? itr->value.GetDouble() : NULL;
    itr = doc.FindMember("currentIteration"); g->currentIteration = (itr != doc.MemberEnd()) ? itr->value.GetUint64() : NULL;
    itr = doc.FindMember("Source");
    if (itr != doc.MemberEnd())
    {
        itr = doc["Source"].FindMember("srclinpos"); g->srclinpos = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
        itr = doc["Source"].FindMember("srcposX"); g->srcposX = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
        itr = doc["Source"].FindMember("srcposY"); g->srcposY = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
        itr = doc["Source"].FindMember("srcposZ"); g->srcposZ = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
        itr = doc["Source"].FindMember("srcFieldComp"); g->srcFieldComp = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
        itr = doc["Source"].FindMember("amplitude"); g->amplitude = (itr != doc.MemberEnd()) ? itr->value.GetDouble() : NULL;
        itr = doc["Source"].FindMember("omega"); g->omega = (itr != doc.MemberEnd()) ? itr->value.GetDouble() : NULL;
        itr = doc["Source"].FindMember("rTime"); g->rTime = (itr != doc.MemberEnd()) ? itr->value.GetDouble() : NULL;
        itr = doc["Source"].FindMember("initPhase"); g->initPhase = (itr != doc.MemberEnd()) ? itr->value.GetDouble() : NULL;
    }
    else
    {
        perror("[rapidJson]: No source defined!");
        return false;
    }
    itr = doc.FindMember("Detector");
    if (itr != doc.MemberEnd())
    {
        itr = doc["Detector"].FindMember("detX"); g->detX = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
        itr = doc["Detector"].FindMember("detY"); g->detY = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
        itr = doc["Detector"].FindMember("detZ"); g->detZ = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
        itr = doc["Detector"].FindMember("detComps"); g->detComps = (itr != doc.MemberEnd()) ? itr->value.GetUint() : NULL;
    }
    else
    {
        perror("[rapidJson]: No detector defined!");
        return false;
    }

    itr = doc.FindMember("Ca");
    if (itr != doc.MemberEnd())
    {
        const Value& a = doc["Ca"]; 
        assert(a.IsArray());
        g->Ca = (float *)calloc(a.Size(), sizeof(float)); //on the spot allocation, should be done somewhere else so it's not too messy with CUDA stuff
        for (SizeType i = 0; i < a.Size(); i++) // rapidjson uses SizeType instead of size_t.
           g->Ca[i] = a[i].GetDouble();
    }
    else
    {
        perror("[rapidJson]: Ca array not found!");
        return false;
    }
    return true;
}


//Writes grid content to a DOM
bool JsonDocument::WriteToDocument(FILE* f, grid *g) 
{
    char writeBuffer[65536];

    Document::AllocatorType& allocator = doc.GetAllocator();
    doc.SetObject();
    Value tmp;
    tmp.SetUint(g->nx);doc.AddMember("nx", tmp, allocator);
    tmp.SetUint(g->ny); doc.AddMember("ny", tmp, allocator);
    tmp.SetUint(g->nz); doc.AddMember("nz", tmp, allocator);
    tmp.SetUint(g->domainSize); doc.AddMember("domainSize", tmp, allocator);
    tmp.SetUint(g->layoutx); doc.AddMember("layoutx", tmp, allocator);
    tmp.SetUint(g->layouty); doc.AddMember("layouty", tmp, allocator);
    tmp.SetUint(g->layoutz); doc.AddMember("layoutz", tmp, allocator);
    tmp.SetInt(g->offset); doc.AddMember("offset", tmp, allocator);
    tmp.SetUint64(g->nt); doc.AddMember("nt", tmp, allocator);
    tmp.SetDouble(g->dx); doc.AddMember("dx", tmp, allocator);
    tmp.SetDouble(g->dy); doc.AddMember("dy", tmp, allocator); 
    tmp.SetDouble(g->dz); doc.AddMember("dz", tmp, allocator);
    tmp.SetUint64(g->currentIteration); doc.AddMember("currentIteration", tmp, allocator);

    tmp.SetArray();
    for (unsigned int i = 0; i < g->domainSize; i++)
        tmp[i].SetDouble(g->Ca[i]);
    doc.AddMember("Ca", tmp, allocator);

    FileWriteStream os(f, writeBuffer, sizeof(writeBuffer));
    Writer<FileWriteStream> writer(os);
    doc.Accept(writer);

    return true;
  /*
    writer.Key("Source");
    writer.StartObject();
    writer.Key("srclinpos"); writer.Uint(g->srclinpos);
    writer.Key("srcposX"); writer.Uint(g->srcposX);
    writer.Key("srcposY"); writer.Uint(g->srcposY);
    writer.Key("srcposZ"); writer.Uint(g->srcposZ);
    writer.Key("srcFieldComp"); writer.Uint(g->srcFieldComp);
    writer.Key("amplitude"); writer.Double(g->amplitude);
    writer.Key("omega"); writer.Double(g->omega);
    writer.Key("rTime"); writer.Double(g->rTime);
    writer.Key("initPhase"); writer.Double(g->initPhase);
    writer.EndObject();
    writer.Key("Detector");
    writer.StartObject();
    writer.Key("detX"); writer.Uint(g->detX);
    writer.Key("detY"); writer.Uint(g->detY);
    writer.Key("detZ"); writer.Uint(g->detZ);
    writer.Key("detComps"); writer.Uint(g->detComps);
    writer.EndObject();
    writer.Key("Ca");
    writer.StartArray();
    for (unsigned i = 0; i < g->domainSize; i++)
        writer.Double(g->Ca[i]);
    writer.EndArray();
    writer.Key("Cb1");
    writer.StartArray();
    for (unsigned i = 0; i < g->domainSize; i++)
        writer.Double(g->Cb1[i]);
    writer.EndArray();
    writer.Key("Cb2");
    writer.StartArray();
    for (unsigned i = 0; i < g->domainSize; i++)
        writer.Double(g->Cb2[i]);
    writer.EndArray();
    writer.Key("Db1");
    writer.StartArray();
    for (unsigned i = 0; i < g->domainSize; i++)
        writer.Double(g->Db1[i]);
    writer.EndArray();
    writer.Key("Db2");
    writer.StartArray();
    for (unsigned i = 0; i < g->domainSize; i++)
        writer.Double(g->Db2[i]);
    writer.EndArray();
    writer.EndObject();
    myfile << s.GetString() << endl;
    myfile.close(); */
}