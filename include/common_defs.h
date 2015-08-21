//common_defs.h: definitions used throughout the software
// Aurelien Duval 2015


#pragma once
#ifndef _COMMON_DEFS_H
#define _COMMON_DEFS_H

#include <stdio.h>
#include <Windows.h>


/*heavily used blocks of memory are allocated using the register keyword*/
/*seems pretty useless as the compiler probably knows better and it's deprecated in C++ 11, "but we never know" :)*/
inline void * cust_alloc(size_t size) {
    register void *value = malloc(size); // allocate a block of memory
    if (value == NULL) // make sure we suceeded in allocating the desired memory
        fprintf(stderr, "Virtual memory exhausted.");
    else
        memset(value, 0, size);
    return value;
}

#endif /*_COMMON_DEFS_H*/