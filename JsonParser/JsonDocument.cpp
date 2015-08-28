//JsonDocument.cpp: Json Document interface to RapidJson
//Aurelien Duval 2015

#include "JsonParser/JsonDocument.h"
#include "rapidJson/filewritestream.h"
#include "rapidJson/prettywriter.h"
#include "common_defs.h"


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
        perror("[JsonParser]: No source defined!");
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
        perror("[JsonParser]: No detector defined!");
        return false;
    }

    itr = doc.FindMember("mat");
    if (itr != doc.MemberEnd())
    {
        const Value& a = doc["mat"];
        assert(a.IsArray());
        g->mat = (unsigned int *)cust_alloc(a.Size()*sizeof(unsigned int));
        for (SizeType i = 0; i < a.Size(); i++) // rapidjson uses SizeType instead of size_t.
            g->mat[i] = a[i].GetUint();
    }
    else
    {
        perror("[JsonParser]: mat array not found!");
        return false;
    }

    itr = doc.FindMember("Nmats"); g->Nmats = (itr != doc.MemberEnd()) ? itr->value.GetDouble() : NULL;

    itr = doc.FindMember("epsilon");
    if (itr != doc.MemberEnd())
    {
        const Value& a = doc["epsilon"];
        assert(a.IsArray());
        g->epsilon = (float *)cust_alloc(a.Size()*sizeof(float));
        for (SizeType i = 0; i < a.Size(); i++) 
            g->epsilon[i] =(float) a[i].GetDouble();
    }
    else
    {
        perror("[JsonParser]: epsilon array not found!");
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
    tmp.SetDouble(g->dt); doc.AddMember("dt", tmp, allocator);
    tmp.SetDouble(g->dx); doc.AddMember("dx", tmp, allocator);
    tmp.SetDouble(g->dy); doc.AddMember("dy", tmp, allocator); 
    tmp.SetDouble(g->dz); doc.AddMember("dz", tmp, allocator);
    tmp.SetUint64(g->currentIteration); doc.AddMember("currentIteration", tmp, allocator);
 

    Value val;
    val.SetObject();
    Value src;
    src.SetUint(g->srclinpos); val.AddMember("srclinpos", src, allocator);
    src.SetUint(g->srcposX); val.AddMember("srcposX", src, allocator);
    src.SetUint(g->srcposY); val.AddMember("srcposY", src, allocator);
    src.SetUint(g->srcposZ); val.AddMember("srcposZ", src, allocator);
    src.SetUint(g->srcFieldComp); val.AddMember("srcFieldComp", src, allocator);
    src.SetDouble(g->amplitude); val.AddMember("amplitude", src, allocator);
    src.SetDouble(g->omega); val.AddMember("omega", src, allocator);
    src.SetDouble(g->rTime); val.AddMember("rTime", src, allocator);
    src.SetDouble(g->initPhase); val.AddMember("initPhase", src, allocator);
    doc.AddMember("Source", val, allocator);

    val.SetObject();
    Value det;
    det.SetUint(g->detX); val.AddMember("detX", det, allocator);
    det.SetUint(g->detY); val.AddMember("detY", det, allocator);
    det.SetUint(g->detZ); val.AddMember("detZ", det, allocator);
    det.SetUint(g->detComps); val.AddMember("detComps", det, allocator);
    doc.AddMember("Detector", val, allocator);

    assert(g->mat != NULL);
    tmp.SetArray();
    for (unsigned int i = 0; i < g->domainSize; i++)
    {
        val.SetUint(g->mat[i]);
        tmp.PushBack(val, doc.GetAllocator());
    }
    doc.AddMember("mat", tmp, allocator);

    assert(g->Nmats >= 1);
    tmp.SetUint(g->Nmats); doc.AddMember("Nmats", tmp, allocator);

    assert(g->epsilon != NULL);
    tmp.SetArray();
    for (unsigned int i = 0; i < g->Nmats; i++)
    {
        val.SetDouble(g->epsilon[i]);
        tmp.PushBack(val, doc.GetAllocator());
    }
    doc.AddMember("epsilon", tmp, allocator);


    FileWriteStream os(f, writeBuffer, sizeof(writeBuffer));
    Writer<FileWriteStream> writer(os);
    doc.Accept(writer);

    return true;

}