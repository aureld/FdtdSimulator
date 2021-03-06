PROJECT(jpeg C)

SET(TARGET_LIB jpeg)

IF(MSVC)
	SET(jconfig "${LIBJPEG_SOURCE_DIR}/jconfig.vc")
ENDIF()

IF (MINGW)
	SET(jconfig "${LIBJPEG_SOURCE_DIR}/jconfig.vc")
ENDIF()

IF (APPLE)
	SET(jconfig "${LIBJPEG_SOURCE_DIR}/jconfig.txt")
ENDIF()


CONFIGURE_FILE(${jconfig} jconfig.h @ONLY)

SET(LIBJPEG_PUBLIC_HEADERS
	${CMAKE_CURRENT_BINARY_DIR}/jconfig.h
	${LIBJPEG_SOURCE_DIR}/jerror.h
	${LIBJPEG_SOURCE_DIR}/jmorecfg.h
	${LIBJPEG_SOURCE_DIR}/jpeglib.h
)
SET(LIBJPEG_PRIVATE_HEADERS
	${LIBJPEG_SOURCE_DIR}/jdct.h
	${LIBJPEG_SOURCE_DIR}/jinclude.h
	${LIBJPEG_SOURCE_DIR}/jmemsys.h
	${LIBJPEG_SOURCE_DIR}/jpegint.h
	${LIBJPEG_SOURCE_DIR}/jversion.h
)
SET(LIBJPEG_SRCS
	${LIBJPEG_SOURCE_DIR}/jaricom.c
	${LIBJPEG_SOURCE_DIR}/jcapimin.c
	${LIBJPEG_SOURCE_DIR}/jcapistd.c
	${LIBJPEG_SOURCE_DIR}/jcarith.c
	${LIBJPEG_SOURCE_DIR}/jccoefct.c
	${LIBJPEG_SOURCE_DIR}/jccolor.c
	${LIBJPEG_SOURCE_DIR}/jcdctmgr.c
	${LIBJPEG_SOURCE_DIR}/jchuff.c
	${LIBJPEG_SOURCE_DIR}/jcinit.c
	${LIBJPEG_SOURCE_DIR}/jcmainct.c
	${LIBJPEG_SOURCE_DIR}/jcmarker.c
	${LIBJPEG_SOURCE_DIR}/jcmaster.c
	${LIBJPEG_SOURCE_DIR}/jcomapi.c
	${LIBJPEG_SOURCE_DIR}/jcparam.c
	${LIBJPEG_SOURCE_DIR}/jcprepct.c
	${LIBJPEG_SOURCE_DIR}/jcsample.c
	${LIBJPEG_SOURCE_DIR}/jctrans.c
	${LIBJPEG_SOURCE_DIR}/jdapimin.c
	${LIBJPEG_SOURCE_DIR}/jdapistd.c
	${LIBJPEG_SOURCE_DIR}/jdarith.c
	${LIBJPEG_SOURCE_DIR}/jdatadst.c
	${LIBJPEG_SOURCE_DIR}/jdatasrc.c
	${LIBJPEG_SOURCE_DIR}/jdcoefct.c
	${LIBJPEG_SOURCE_DIR}/jdcolor.c
	${LIBJPEG_SOURCE_DIR}/jddctmgr.c
	${LIBJPEG_SOURCE_DIR}/jdhuff.c
	${LIBJPEG_SOURCE_DIR}/jdinput.c
	${LIBJPEG_SOURCE_DIR}/jdmainct.c
	${LIBJPEG_SOURCE_DIR}/jdmarker.c
	${LIBJPEG_SOURCE_DIR}/jdmaster.c
	${LIBJPEG_SOURCE_DIR}/jdmerge.c
	${LIBJPEG_SOURCE_DIR}/jdpostct.c
	${LIBJPEG_SOURCE_DIR}/jdsample.c
	${LIBJPEG_SOURCE_DIR}/jdtrans.c
	${LIBJPEG_SOURCE_DIR}/jerror.c
	${LIBJPEG_SOURCE_DIR}/jfdctflt.c
	${LIBJPEG_SOURCE_DIR}/jfdctfst.c
	${LIBJPEG_SOURCE_DIR}/jfdctint.c
	${LIBJPEG_SOURCE_DIR}/jidctflt.c
	${LIBJPEG_SOURCE_DIR}/jidctfst.c
	${LIBJPEG_SOURCE_DIR}/jidctint.c
	${LIBJPEG_SOURCE_DIR}/jmemmgr.c
	${LIBJPEG_SOURCE_DIR}/jmemnobs.c
	${LIBJPEG_SOURCE_DIR}/jquant1.c
	${LIBJPEG_SOURCE_DIR}/jquant2.c
	${LIBJPEG_SOURCE_DIR}/jutils.c
)

# Dependent include files
INCLUDE_DIRECTORIES(${LIBJPEG_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

IF(MSVC)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Gy")
	ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS -D_LIB)
	SET(CMAKE_DEBUG_POSTFIX "d")
ENDIF(MSVC)

ADD_LIBRARY(${TARGET_LIB} STATIC ${LIBJPEG_SRCS} ${LIBJPEG_PRIVATE_HEADERS} ${LIBJPEG_PUBLIC_HEADERS})

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
	install(FILES ${LIBJPEG_PUBLIC_HEADERS} DESTINATION ${INCLUDE_OUTPUT_PATH})
ENDIF()

