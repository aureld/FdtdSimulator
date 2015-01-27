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
        snapshot.cpp
        stepping.cpp
        update.cpp
)

# Dependent include files
#INCLUDE_DIRECTORIES(${LIBTIFF_PUBLIC_HEADERS} ${CMAKE_CURRENT_BINARY_DIR})
#INCLUDE_DIRECTORIES("${LIFTIFF_SOURCE_DIR}/libtiff")
#INCLUDE_DIRECTORIES("${MOVIELIB_SOURCE_DIR}")
#INCLUDE_DIRECTORIES(${VTK_USE_FILE})
find_package(TIFF)


ADD_LIBRARY(FDTDCoreLib ${FDTDCORELIB_SRCS} ${FDTDCORELIB_HEADERS})



