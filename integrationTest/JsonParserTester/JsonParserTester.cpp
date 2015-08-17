// JsonParserTester.cpp : Black box testing of JsonParser lib
//Aurelien Duval 2015


#include "JsonParser\FileIO.h"
#include "JsonParser\JSonDocument.h"
#include "JsonParser\JSonParser.h"
#include "gtest/gtest.h"
#include <process.h>

TEST(JsonParserTester, CompareFiles)
{
    grid *g;
    g = (grid*)calloc(1, sizeof(grid));


    JsonDocument *doc = new JsonDocument();
    FileIO *fio = new FileIO();
    JsonParser *parser = new JsonParser(doc, fio);
    //we run the parser in and out 
    parser->ParseJsonFile("ValidTestJson.txt", g);
    parser->ExportToJson("ExportedTestJson.txt", g);

    std::string path = "C:\\code\\fdtdSimC\\build\\x64\\bin\\";
    std::string param1 = path + "ValidTestJson.txt";
    std::string param2 = path + "ExportedTestJson.txt";

    //we compare the input and output files.
    //we might run into troubles when comparing double outputs or arrays... to be improved
    int ret = _spawnlp(P_WAIT, "fc.exe", "fc.exe", param1.c_str(), param2.c_str() , NULL);
    EXPECT_EQ(0, ret);
}

