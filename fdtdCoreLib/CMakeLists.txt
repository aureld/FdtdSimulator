PROJECT(FDTDCoreLib CXX)

SET(TARGET_LIB fdtdcorelib)

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
INCLUDE_DIRECTORIES("${LIFTIFF_SOURCE_DIR}/libtiff")
INCLUDE_DIRECTORIES("${MOVIELIB_SOURCE_DIR}")
INCLUDE_DIRECTORIES(${VTK_USE_FILE})

IF(MSVC)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Gy")
        ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS -D_LIB)
        SET(CMAKE_DEBUG_POSTFIX "d")
ENDIF(MSVC)

ADD_LIBRARY(${TARGET_LIB} STATIC ${FDTDCORELIB_SRCS} ${FDTDCORELIB_HEADERS})

# Runtime equals DLL-files
# Archive equals static libraries
# Library equals module libraries
IF(NOT SKIP_INSTALL_LIBRARIES AND NOT SKIP_INSTALL_ALL )
        install(TARGETS ${TARGET_LIB}
                RUNTIME DESTINATION ${EXECUTABLE_OUTPUT_PATH}
                LIBRARY DESTINATION ${LIBRARY_OUTPUT_PATH}
                ARCHIVE DESTINATION ${ARCHIVE_OUTPUT_PATH})
ENDIF()
IF(NOT SKIP_INSTALL_HEADERS AND NOT SKIP_INSTALL_ALL )
        install(FILES ${FDTDCORELIB_HEADERS} DESTINATION ${INCLUDE_OUTPUT_PATH})
ENDIF()



