set( HEADERS my_lapack.h util.h Mat.h err.h)

if ( WIN32 )
    set( FLAGS_DEBUG /DEBUG /Od /openmp ) 
    set( FLAGS_RELEASE /O2 /openmp )
else()
    set (FLAGS_DEBUG -g -fopenmp )
    set( FLAGS_RELEASE -O3 -fopenmp )
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_definitions ( "-DLAHPC_DEBUG" )
    endif()

function(create_my_lapack_lib libname )
    add_library( ${libname} 
        ${libname}.cpp
        util.cpp
        Mat.cpp
        ${HEADERS} )
    
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
    
    target_compile_options( 
        ${libname}
        PRIVATE
            "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
            "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
    )
endfunction()

create_my_lapack_lib("my_lapack_seq")
create_my_lapack_lib("my_lapack_omp")

