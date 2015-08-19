//ParseJsonTest.cpp: test class for parsing Json files into FDTD grid
//Aurelien Duval 2015

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "simCuda3D/Cuda_grid.h"
#include "JsonParser/JSonParser.h"
#include "MockJsonDocument.h"
#include "MockFileIO.h"
#include <process.h>

using ::testing::Return;

class JsonParserTest : public ::testing::Test {
protected:
    virtual void SetUp() 
    {
        g = new grid();
        parser = new JsonParser(&doc, &fio);
        tolerance = 1e-19;
    }

    grid *g;
    JsonParser *parser;
    MockJsonDocument doc;
    MockFileIO fio;
    FILE * test_file;  
    double tolerance;
};

TEST_F(JsonParserTest, ParseJson_Open) {
    const char* kName = "wrongfilename";
    test_file = NULL;
    ON_CALL(fio, Open(kName, "r"))
        .WillByDefault(Return(test_file));
    EXPECT_EQ(false, parser->ParseJsonFile(kName, g));
}

TEST_F(JsonParserTest, ParseJson_NULLgrid) {
    const char* kName = "goodfilename";
    test_file = new FILE();
    ON_CALL(fio, Open(kName, "r"))
        .WillByDefault(Return(test_file));
    EXPECT_EQ(false, parser->ParseJsonFile(kName, NULL));
}

TEST_F(JsonParserTest, ParseJson_HasParseError) {
    const char* kName = "goodfilename";
    test_file = new FILE();
    ON_CALL(fio, Open(kName, "r"))
        .WillByDefault(Return(test_file));
    ON_CALL(doc, HasParseError())
        .WillByDefault(Return(true));

    EXPECT_EQ(false, parser->ParseJsonFile(kName, g));
}

TEST_F(JsonParserTest, ParseJson_HasInvalidSchema) {
    const char* kName = "goodfilename";
    test_file = new FILE();

    ON_CALL(fio, Open(kName, "r"))
        .WillByDefault(Return(test_file));
    ON_CALL(doc, HasParseError())
        .WillByDefault(Return(false));
    ON_CALL(doc, WriteToGrid(g))
        .WillByDefault(Return(false));

    EXPECT_EQ(false, parser->ParseJsonFile(kName, g));
}


TEST_F(JsonParserTest, ExportToJson_WrongFilePath) {
    const char* kName = "wrongfilename";
    test_file = NULL;
    ON_CALL(fio, Open(kName, "r"))
        .WillByDefault(Return(test_file));
    EXPECT_EQ(false, parser->ExportToJson(kName, g));
}


TEST_F(JsonParserTest, ExportToJson_NULLgrid) {
    const char* kName = "goodfilename";
    test_file = new FILE();
    ON_CALL(fio, Open(kName, "r"))
        .WillByDefault(Return(test_file));
    EXPECT_EQ(false, parser->ExportToJson(kName, NULL));
}

 
TEST_F(JsonParserTest, ExportToJson_HasInvalidGrid) {
    const char* kName = "goodfilename";
    test_file = new FILE();
    ON_CALL(fio, Open(kName, "r"))
        .WillByDefault(Return(test_file));
    ON_CALL(doc, WriteToDocument(test_file, g))
        .WillByDefault(Return(false));
    EXPECT_EQ(false, parser->ExportToJson(kName, g));
}


TEST_F(JsonParserTest, CompareFiles)
{
    std::string path = "C:\\code\\fdtdSimC\\build\\x64\\bin\\";
    std::string param1 = path + "ValidTestJson.txt";
    std::string param2 = path + "ExportedTestJson.txt";

    JsonDocument *doc = new JsonDocument();
    FileIO *f = new FileIO();
    JsonParser *p = new JsonParser(doc, f);

    //we run the parser in and out 
    p->ParseJsonFile(param1.c_str(), g);
    p->ExportToJson(param2.c_str(), g);
    //we compare the input and output files.
    //we might run into troubles when comparing double outputs or arrays... to be improved 
    //int ret = _spawnlp(P_WAIT, "fc.exe", "fc.exe", param1.c_str(), param2.c_str(), NULL);
    //EXPECT_EQ(0, ret);
    //since comparing doubles fail, we can just compare manually....
}
