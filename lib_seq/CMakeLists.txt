set( HEADERS my_lapack.h util.h Mat.h err.h)
set( FLAGS_DEBUG /DEBUG /Od ) 
set( FLAGS_RELEASE /O2 )

add_library( my_lapack_seq 
    my_lapack.cpp
    util.cpp
    Mat.cpp
    ${HEADERS} )

target_include_directories(
    my_lapack_seq
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/perf
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions ( "-DLAHPC_DEBUG" )
endif()

target_compile_features(
    my_lapack_seq
    PUBLIC
        cxx_std_11
)

target_compile_options( 
    my_lapack_seq
    PRIVATE
        "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
        "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
)