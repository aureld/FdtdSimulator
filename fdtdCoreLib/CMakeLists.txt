cmake_minimum_required(VERSION 3.0)

PROJECT(FDTDCoreLib CXX)

set (FDTDCORELIB_HEADERS
        fdtd-macros.h
        fdtd-protos.h
        fdtd-alloc.h        
        fdtd-structs.h
)

set (FDTDCORELIB_SRCS
        ricker.cpp
        gridinit.cpp
        stepping.cpp
        update.cpp
)

SET(FDTDCORELIB_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Path where to find FDTDCoreLib source")

# Dependent include files
INCLUDE_DIRECTORIES("${MOVIELIB_SOURCE_DIR}")


ADD_LIBRARY(FDTDCoreLib ${FDTDCORELIB_SRCS} ${FDTDCORELIB_HEADERS})



