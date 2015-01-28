cmake_minimum_required (VERSION 3.0)

project (fdtdCore)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (MSVC)
    string(REPLACE "/W3" "/W4" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}) 
    string(REPLACE "/W3" "/W4" CMAKE_C_FLAGS ${CMAKE_C_FLAGS}) 
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /openmp")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    IF(CMAKE_SIZEOF_VOID_P MATCHES 8)
        #64-bit compiler
        string(REPLACE "/DWIN32" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}) 
        string(REPLACE "/DWIN32" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS}) 
        add_definitions(/D_WIN64)
    ENDIF()
endif(MSVC)


message ("cxx Flags: " ${CMAKE_CXX_FLAGS})


# Generate folder name for install
IF(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
    SET(BITS "x64")
ELSEIF (${CMAKE_SIZEOF_VOID_P} EQUAL 4)
    SET(BITS "x86")
ENDIF()

SET(dirname "${BITS}")

SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${dirname}/bin")
SET(ARCHIVE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${dirname}/lib")
SET(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${dirname}/lib")
SET(INCLUDE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${dirname}/include")
link_directories(${LIBRARY_OUTPUT_PATH})

if (MSVC)
    foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
        string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
        set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${EXECUTABLE_OUTPUT_PATH} )
        set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${LIBRARY_OUTPUT_PATH} )
        set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${ARCHIVE_OUTPUT_PATH} )
    endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
endif(MSVC)

#Set directories
#MODIFY IF NEEDED
SET(VTK_ROOT "C:/Program Files/VTK")
SET(VTK_DIR "${VTK_ROOT}/lib/cmake/vtk-6.2")
SET(VTK_BIN "${VTK_ROOT}/bin")





#add_subdirectory(3rdParty)
add_subdirectory(MovieLib)
add_subdirectory(fdtdCoreLib)
add_subdirectory(runFDTD)

#SET_PROPERTY(TARGET zlib PROPERTY FOLDER "extLibs")
#SET_PROPERTY(TARGET jpeg PROPERTY FOLDER "extLibs")
#SET_PROPERTY(TARGET libtiff PROPERTY FOLDER "extLibs")
SET_PROPERTY(TARGET FDTDCoreLib PROPERTY FOLDER "Libs")
SET_PROPERTY(TARGET MovieLib PROPERTY FOLDER "Libs")





