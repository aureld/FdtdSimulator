cmake_minimum_required (VERSION 3.0)

project (fdtdCore)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")

message ("cxx Flags: " ${CMAKE_CXX_FLAGS})

SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${dirname}/bin")
SET(ARCHIVE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${dirname}/lib")
SET(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/lib")
SET(INCLUDE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${dirname}/include")

#Set directories
#MODIFY IF NEEDED
#SET(LIB_SRC_PATH "C:/Projects/FDTDsimC/3rdParty/src")
#SET(ZLIB_SOURCE_DIR "${LIB_SRC_PATH}/zlib-1.2.8" CACHE PATH "Path where to find ZLIB source")
#SET(LIBJPEG_SOURCE_DIR "${LIB_SRC_PATH}/jpeg-9a" CACHE PATH "Path where to find LIBJPEG source")
#SET(LIFTIFF_SOURCE_DIR "${LIB_SRC_PATH}/tiff-4.0.3" CACHE PATH "Path where to find LIBTIFF source")
#SET(MOVIELIB_SOURCE_DIR "${CMAKE_SOURCE_DIR}/MovieLib" CACHE PATH "Path where to find MovieLib source")
#SET(FDTDCORELIB_SOURCE_DIR "${CMAKE_SOURCE_DIR}/fdtdCoreLib" CACHE PATH "Path where to find FDTDCoreLib source")

SET(VTK_DIR "C:/Program Files/VTK/lib/cmake/vtk-6.2")


FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

add_subdirectory(3rdParty)
#add_subdirectory(fdtdCoreLib)
#add_subdirectory(MovieLib)
#add_subdirectory(runFDTD)

#target_link_libraries(runFDTD FDTDCoreLib)

SET_PROPERTY(TARGET zlib PROPERTY FOLDER "Libs")
SET_PROPERTY(TARGET jpeg PROPERTY FOLDER "Libs")
SET_PROPERTY(TARGET libtiff PROPERTY FOLDER "Libs")



