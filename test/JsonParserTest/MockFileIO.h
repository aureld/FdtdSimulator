//MockFileIO.h: Mock implementation of file IO operations 
//Aurelien Duval 2015

#include "gmock/gmock.h"
#include "JsonParser/FileIO.h"

class MockFileIO : public IFileIO {
public:
    virtual ~MockFileIO() { }

    MOCK_METHOD2(Open, FILE*(const char* filename, const char* mode));
    MOCK_METHOD4(Write, size_t(const void* data,
        size_t size, size_t num, FILE* file));
    MOCK_METHOD1(Close, int(FILE* file));
};


