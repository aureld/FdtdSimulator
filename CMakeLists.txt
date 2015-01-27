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
SET(LIB_SRC_PATH "/Users/aurel/code/fdtdsimc/3rdParty/src" CACHE PATH "Path where to find 3rd party libs")

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



