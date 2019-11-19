## COMMON

if ( WIN32 )
    set( FLAGS_DEBUG /DEBUG /Od ) 
    set( FLAGS_RELEASE /O2 )
else()
    set (FLAGS_DEBUG -g)
    set( FLAGS_RELEASE -O3 )
endif()

############################################################
######################   SEQUENTIAL   ######################
############################################################

########## driver
function(make_driver my_lapack_lib)

    add_executable(
        "driver_${my_lapack_lib}"
        ${CMAKE_CURRENT_SOURCE_DIR}/driver.cpp
    )
    
    target_include_directories(
        "driver_${my_lapack_lib}"
        PRIVATE
        "${PROJECT_SOURCE_DIR}/lib"
    )
    
    target_compile_options( 
        "driver_${my_lapack_lib}"
        PRIVATE
            "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
            "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
    )
    
    set_target_properties(
        "driver_${my_lapack_lib}"
        PROPERTIES CXX_STANDARD 11
    )
    
    target_link_libraries(
        "driver_${my_lapack_lib}"
        PRIVATE 
            ${my_lapack_lib}
    )

endfunction()

make_driver("my_lapack_seq")
#make_driver("my_lapack_omp")


########## test_valid_seq
add_executable(
    test_valid_seq
    ${CMAKE_CURRENT_SOURCE_DIR}/test_valid.cpp
)

target_include_directories(
    test_valid_seq
    PRIVATE
        "${PROJECT_SOURCE_DIR}/lib"
        "${PROJECT_SOURCE_DIR}/algonum/include"
)

find_library(
    ALGONUM_LIBRARY
    NAMES libalgonum.so libalgonum
    PATHS "${PROJECT_SOURCE_DIR}/algonum/lib" "${PROJECT_SOURCE_DIR}/algonum/lib/libalgonum.so"
)

target_compile_options( 
    test_valid_seq
    PRIVATE
    "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
    )

set_target_properties(
    test_valid_seq
    PROPERTIES CXX_STANDARD 11
    )
        
target_link_libraries(
    test_valid_seq
    PRIVATE 
        my_lapack_seq
        #ALGONUM_LIBRARY
)


########## test_perf_seq
add_executable(
    test_perf_seq
    ${CMAKE_CURRENT_SOURCE_DIR}/test_perf.cpp
)

target_include_directories(
    test_perf_seq
    PRIVATE
    "${PROJECT_SOURCE_DIR}/lib"
)

target_compile_options( 
    test_perf_seq
    PRIVATE
    "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
    )

set_target_properties(
    test_perf_seq
    PROPERTIES CXX_STANDARD 11
    )
        
target_link_libraries(
    test_perf_seq
    PRIVATE 
        my_lapack_seq
        #ALGONUM_LIBRARY
)