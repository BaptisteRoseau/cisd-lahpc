## COMMON

set ( FLAGS_DEBUG /DEBUG /Od ) 
set ( FLAGS_RELEASE /O2 )

############################################################
######################   SEQUENTIAL   ######################
############################################################

########## driver_seq
add_executable(
    driver_seq
    ${CMAKE_CURRENT_SOURCE_DIR}/driver.cpp
)

target_include_directories(
    driver_seq
    PRIVATE
    "${PROJECT_SOURCE_DIR}/lib_seq"
)

target_compile_options( 
    driver_seq
    PRIVATE
        "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
        "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
)

set_target_properties(
    driver_seq
    PROPERTIES CXX_STANDARD 11
)

target_link_libraries(
    driver_seq
    PRIVATE 
        my_lapack_seq
)

########## test_valid_seq
add_executable(
    test_valid_seq
    ${CMAKE_CURRENT_SOURCE_DIR}/test_valid.cpp
)

target_include_directories(
    test_valid_seq
    PRIVATE
        "${PROJECT_SOURCE_DIR}/lib_seq"
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
    -L "${PROJECT_SOURCE_DIR}/algonum/lib"
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
    "${PROJECT_SOURCE_DIR}/lib_seq"
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


############################################################
######################     OPENMP     ######################
############################################################


########## driver_omp
add_executable(
    driver_omp
    ${CMAKE_CURRENT_SOURCE_DIR}/driver.cpp
)

target_include_directories(
    driver_omp
    PRIVATE
    "${PROJECT_SOURCE_DIR}/lib_omp"
)

target_compile_options( 
    driver_omp
    PRIVATE
        "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
        "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
)

set_target_properties(
    driver_omp
    PROPERTIES CXX_STANDARD 11
)

target_link_libraries(
    driver_omp
    PRIVATE 
        my_lapack_omp
)

########## test_valid_omp
add_executable(
    test_valid_omp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_valid.cpp
)

target_include_directories(
    test_valid_omp
    PRIVATE
        "${PROJECT_SOURCE_DIR}/lib_omp"
        "${PROJECT_SOURCE_DIR}/algonum/include"
)


find_library(
    ALGONUM_LIBRARY
    NAMES libalgonum.so libalgonum
    PATHS "${PROJECT_SOURCE_DIR}/algonum/lib" "${PROJECT_SOURCE_DIR}/algonum/lib/libalgonum.so"
)

target_compile_options( 
    test_valid_omp
    PRIVATE
    "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
    -L "${PROJECT_SOURCE_DIR}/algonum/lib"
    )

set_target_properties(
    test_valid_omp
    PROPERTIES CXX_STANDARD 11
    )
        
target_link_libraries(
    test_valid_omp
    PRIVATE 
        my_lapack_omp
        #ALGONUM_LIBRARY
)


######### test_perf_omp
add_executable(
    test_perf_omp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_perf.cpp
)

target_include_directories(
    test_perf_omp
    PRIVATE
    "${PROJECT_SOURCE_DIR}/lib_omp"
)

target_compile_options( 
    test_perf_omp
    PRIVATE
    "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
    )

set_target_properties(
    test_perf_omp
    PROPERTIES CXX_STANDARD 11
    )
        
target_link_libraries(
    test_perf_omp
    PRIVATE 
        my_lapack_omp
        #ALGONUM_LIBRARY
)