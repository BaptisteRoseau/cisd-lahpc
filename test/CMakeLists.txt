add_executable(
    driver
    ${CMAKE_CURRENT_SOURCE_DIR}/driver.cpp
)

target_include_directories(
    driver
    PRIVATE
        "${PROJECT_SOURCE_DIR}/lib"
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
        my_lapack
)