// JsonParserTester.cpp : Black box testing of JsonParser lib
//Aurelien Duval 2015


#include "JsonParser\FileIO.h"
#include "JsonParser\JSonDocument.h"
#include "JsonParser\JSonParser.h"
#include "gtest/gtest.h"
#include <process.h>


class JsonParserTester : public ::testing::Test {
protected:
    virtual void SetUp()
    {
        g = new grid();
        d = new JsonDocument();
        fio = new FileIO();
        parser = new JsonParser(d, fio);
    }

    grid *g;
    JsonParser *parser;
    IJsonDocument *d;
    IFileIO *fio;
    FILE * test_file;
    double tolerance;
};


TEST_F(JsonParserTester, CompareFiles)
{
    std::string path = "C:\\code\\fdtdSimC\\build\\x64\\bin\\";
    std::string param1 = path + "ValidTestJson.txt";
    std::string param2 = path + "ExportedTestJson.txt";

    //we run the parser in and out 
    parser->ParseJsonFile(param1.c_str(), g);
    parser->ExportToJson(param2.c_str(), g);


    //we compare the input and output files.
    //we might run into troubles when comparing double outputs or arrays... to be improved
    int ret = _spawnlp(P_WAIT, "fc.exe", "fc.exe", param1.c_str(), param2.c_str() , NULL);
    EXPECT_EQ(0, ret);
}

