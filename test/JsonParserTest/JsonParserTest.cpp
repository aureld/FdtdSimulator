//ParseJsonTest.cpp: test class for parsing Json files into FDTD grid
//Aurelien Duval 2015

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "simCuda3D/Cuda_grid.h"
#include "JsonParser/JSonParser.h"
#include "MockJsonDocument.h"
#include "MockFileIO.h"

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
    Document d;
    MockFileIO fio;
    FILE * test_file;  
    double tolerance;
};

TEST_F(JsonParserTest, ParseJson_WrongFilePath) {
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

