set (FDTD_CORE_H
        "fdtd-macros.h"
        "fdtd-protos.h"
        "fdtd-alloc.h"
        "fdtd-structs.h")
set (FDTD_CORE_CPP
        "gridinit.cpp"
        "ricker.cpp"
        "snapshot.cpp"
        "stepping.cpp"
        "update.cpp")

source_group("header files" FILES ${FDTD_CORE_H})
source_group("source files" FILES ${FDTD_CORE_CPP})

include_directories(.)

add_library(fdtdCoreLib STATIC  ${FDTD_CORE_H}
                                ${FDTD_CORE_CPP})

set_property(TARGET fdtdCoreLib PROPERTY FOLDER "Libraries")

