### COMMON
set( COMMON_HEADERS my_lapack.h util.h Mat.h err.h)

if ( WIN32 )
    set( FLAGS_DEBUG /DEBUG /Od ) 
    set( FLAGS_RELEASE /O2 )
else()
    set (FLAGS_DEBUG -g -Wall -Wextra -Wno-unused-parameter)
    set( FLAGS_RELEASE -O3 -Wall -Wextra -Wno-unused-parameter)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions ( "-DLAHPC_DEBUG" )
endif()

### SPECIFIC LIB
function(my_lapack_lib_common libname )
    add_library( ${libname} 
        ${libname}.cpp
        util.cpp
        Mat.cpp
        ${COMMON_HEADERS} )
    
    target_include_directories(
        ${libname}
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/perf
    )
    
    target_compile_features(
        ${libname}
        PUBLIC
            cxx_std_11
    )

    if ( WIN32 )
        target_compile_options( 
            ${libname}
            PUBLIC
                "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
                "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
                "-D_${libname}"
        )
    else()
        target_compile_options( 
            ${libname}
            PUBLIC
                "${FLAGS_DEBUG}"
                "${FLAGS_RELEASE}"
                "-D_${libname}"
        )
    endif()

endfunction()

### OpenMP
function(link_omp libname)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${libname} PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(FATAL_ERROR "OpenMP not found !")
    endif()
endfunction()

### MPI
function(link_mpi libname)
    find_package(MPI)
    if (MPI_CXX_FOUND)
        target_link_libraries(${libname} PUBLIC MPI::MPI_CXX ${MPI_CXX_LINK_FLAGS})
    else()
        message(FATAL_ERROR "MPI not found !")
    endif()

    target_include_directories(${libname}
        PUBLIC ${MPI_CXX_INCLUDE_DIRS}
    )

    target_compile_options(${libname}
        PUBLIC
            ${MPI_CXX_COMPILER_FLAGS}
    )

    #set(MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS} -n 4)

endfunction()


### ALL-IN-ONE LIBRARY
add_library( my_lapack_all
    my_lapack_omp.cpp
    my_lapack_seq.cpp
    my_lapack_mpi.cpp
    util.cpp
    Mat.cpp
    Summa.cpp
    ${COMMON_HEADERS}
    Summa.hpp)

target_include_directories(
    my_lapack_all
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/perf

)

target_compile_features(
    my_lapack_all
    PUBLIC
        cxx_std_11
)

if ( WIN32 )
    target_compile_options( 
        my_lapack_all
        PUBLIC
            "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
            "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
            "-D_my_lapack_all"
    )
else()
    target_compile_options( 
        my_lapack_all
        PUBLIC
            "${FLAGS_DEBUG}"
            "${FLAGS_RELEASE}"
            "-D_my_lapack_all"
    )
endif()

link_omp("my_lapack_all")
link_mpi("my_lapack_all")


# CREATING LIBRARIES

my_lapack_lib_common("my_lapack_seq")

my_lapack_lib_common("my_lapack_omp")
link_omp("my_lapack_omp")

my_lapack_lib_common("my_lapack_mpi")
link_mpi("my_lapack_mpi")