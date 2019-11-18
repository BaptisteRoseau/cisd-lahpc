# DRIVER
add_executable(
    driver
    ${CMAKE_CURRENT_SOURCE_DIR}/driver.cpp
)

target_include_directories(
    driver
    PRIVATE
    "${PROJECT_SOURCE_DIR}/lib_ocv"
    "${PROJECT_SOURCE_DIR}/lib_seq"
)

set ( FLAGS_DEBUG /DEBUG /Od ) 
set ( FLAGS_RELEASE /O2 )

target_compile_options( 
    driver
    PRIVATE
        "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
        "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
)

set_target_properties(
    driver
    PROPERTIES CXX_STANDARD 11
)

target_link_libraries(
    driver
    PRIVATE 
        my_lapack_seq
)

# TEST_VALID
add_executable(
    test_valid
    ${CMAKE_CURRENT_SOURCE_DIR}/test_valid.cpp
)

target_include_directories(
    test_valid
    PRIVATE
        "${PROJECT_SOURCE_DIR}/lib_ocv"
        "${PROJECT_SOURCE_DIR}/lib_seq"
        "${PROJECT_SOURCE_DIR}/algonum/include"
)


set ( FLAGS_DEBUG /DEBUG /Od ) 
set ( FLAGS_RELEASE /O2 )

find_library(
    ALGONUM_LIBRARY
    NAMES libalgonum.so libalgonum
    PATHS "${PROJECT_SOURCE_DIR}/algonum/lib" "${PROJECT_SOURCE_DIR}/algonum/lib/libalgonum.so"
)

target_compile_options( 
    test_valid
    PRIVATE
    "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
    -L "${PROJECT_SOURCE_DIR}/algonum/lib"
    )

set_target_properties(
    test_valid
    PROPERTIES CXX_STANDARD 11
    )
        
target_link_libraries(
    test_valid
    PRIVATE 
        my_lapack_seq
        #ALGONUM_LIBRARY
)


# TEST_PERF
add_executable(
    test_perf
    ${CMAKE_CURRENT_SOURCE_DIR}/test_perf.cpp
)

target_include_directories(
    test_perf
    PRIVATE
    "${PROJECT_SOURCE_DIR}/lib_ocv"
    "${PROJECT_SOURCE_DIR}/lib_seq"
)

set ( FLAGS_DEBUG /DEBUG /Od ) 
set ( FLAGS_RELEASE /O2 )

target_compile_options( 
    test_perf
    PRIVATE
    "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
    )

set_target_properties(
    test_perf
    PROPERTIES CXX_STANDARD 11
    )
        
target_link_libraries(
    test_perf
    PRIVATE 
        my_lapack_seq
        #ALGONUM_LIBRARY
)