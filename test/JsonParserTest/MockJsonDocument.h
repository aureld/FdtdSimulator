//MockJsonDocument.cpp: Mock implementation of the JsonDocument interface
//Aurelien Duval 2015

#include "gmock/gmock.h"
#include "JsonParser/JSonDocument.h" 
#include "rapidJson/filewritestream.h"

class MockJsonDocument : public IJsonDocument {
public:
    virtual ~MockJsonDocument() { }
    MOCK_METHOD1(ParseStream, Document&(FileReadStream is));
    MOCK_METHOD0(HasParseError, bool());
    MOCK_METHOD1(WriteToGrid, bool(grid *g));
    MOCK_METHOD2(WriteToDocument, bool(FILE *f, grid *g));

};