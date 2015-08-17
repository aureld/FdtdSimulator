//FileIO.h: file I/O functionalities abstracted from the OS so it's unit testable
//Aurelien Duval 2015
#pragma once
#ifndef _FILEIO_H_
#define _FILEIO_H_

#include <stdio.h>

class IFileIO {
public:
    virtual ~IFileIO() {}

    virtual FILE* Open(const char* filename, const char* mode) = 0;
    virtual size_t Write(const void* data, size_t size, size_t num, FILE* file) = 0;
    virtual int Close(FILE* file) = 0;
};


class FileIO : public IFileIO {
public:
    virtual FILE* Open(const char* filename, const char* mode) {
        return fopen(filename, mode);
    }

    virtual size_t Write(const void* data, size_t size, size_t num, FILE* file) {
        return fwrite(data, size, num, file);
    }

    virtual int Close(FILE* file) {
        return fclose(file);
    }
};



#endif // _FILEIO_H_
