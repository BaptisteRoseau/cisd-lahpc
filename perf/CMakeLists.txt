if ( WIN32 )
    set( FLAGS_DEBUG /DEBUG /Od ) 
    set( FLAGS_RELEASE /O2 )
else()
    set (FLAGS_DEBUG -g)
    set( FLAGS_RELEASE -O3 )
endif()

function(make_benchmark my_lapack_lib)

    add_executable(
        benchmark
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cpp
    )
    
    target_include_directories(
        benchmark
        PRIVATE
            "${PROJECT_SOURCE_DIR}/lib"
    )
    
    target_compile_options( 
        benchmark
        PRIVATE
            "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
            "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
    )
    
    set ( FLAGS_LINKER_DEBUG /PROFILE)
    
    target_link_options (
        benchmark
        PRIVATE
            "$<$<CONFIG:Debug>:${FLAGS_LINKER_DEBUG}>"   
    )
    
    set_target_properties(
        benchmark
        PROPERTIES CXX_STANDARD 11
    )

    target_link_libraries(
        benchmark
        PRIVATE 
            ${my_lapack_lib}
    )

endfunction()

make_benchmark("my_lapack_seq")
