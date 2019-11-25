set( HEADERS my_lapack.h util.h Mat.h err.h)

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

    if ( WIN32 )
        target_compile_options( 
            ${libname}
            PUBLIC
                "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
                "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
        )
    else()
        target_compile_options( 
            ${libname}
            PUBLIC
                "${FLAGS_DEBUG}"
                "${FLAGS_RELEASE}"
        )
    endif()

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${libname} PUBLIC OpenMP::OpenMP_CXX)
    endif()
endfunction()

create_my_lapack_lib("my_lapack_seq")
create_my_lapack_lib("my_lapack_omp")

