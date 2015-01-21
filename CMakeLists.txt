cmake_minimum_required (VERSION 2.8.11)

project (fdtdCore)

#set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")

#message ("cxx Flags: " ${CMAKE_CXX_FLAGS})

SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${dirname}/bin")
SET(ARCHIVE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${dirname}/lib")
SET(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/lib")
SET(INCLUDE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${dirname}/include")

#Set directories
#MODIFY IF NEEDED
SET(LIB_SRC_PATH "/Users/aurel/code/fdtdsimc/3rdParty/src" CACHE PATH "Path where to find 3rd party libs")

SET(ZLIB_SOURCE_DIR "${LIB_SRC_PATH}/zlib-1.2.8" CACHE PATH "Path where to find ZLIB source")
#SET(LIBPNG_SOURCE_DIR CACHE PATH "Path where to find LIBPNG source")
SET(LIBJPEG_SOURCE_DIR "${LIB_SRC_PATH}/jpeg-9a" CACHE PATH "Path where to find LIBJPEG source")
SET(LIFTIFF_SOURCE_DIR "${LIB_SRC_PATH}/tiff-4.0.3" CACHE PATH "Path where to find LIBTIFF source")
#SET(FREETYPE_SOURCE_DIR CACHE PATH "Path where to find FREETYPE source")
#SET(GLUT_SOURCE_DIR CACHE PATH "Path where to find GLUT source")
#SET(GIFLIB_SOURCE_DIR CACHE PATH "Path where to find GIFLIB source")
#SET(MINIZIP_SOURCE_DIR CACHE PATH "Path where to find MINIZIP source")

SET(MOVIELIB_SOURCE_DIR "${CMAKE_SOURCE_DIR}/MovieLib" CACHE PATH "Path where to find MovieLib source")

#SET(VTK_DIR "C:/Program Files/VTK/lib/cmake/vtk-6.2")
FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

add_subdirectory(3rdParty)
add_subdirectory(fdtdCoreLib)
add_subdirectory(MovieLib)
#add_subdirectory(runFDTD)
