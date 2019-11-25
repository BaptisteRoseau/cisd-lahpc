if ( WIN32 )
    set( FLAGS_DEBUG /DEBUG /Od /openmp) 
    set( FLAGS_RELEASE /O2 /openmp )
else()
    set (FLAGS_DEBUG -g  -fopenmp )
    set( FLAGS_RELEASE -O3  -fopenmp )
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
    
    if ( WIN32 )
        target_compile_options( 
            benchmark
            PRIVATE
                "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
                "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
        )
    else()
        target_compile_options( 
            benchmark
            PRIVATE
                "${FLAGS_DEBUG}"
                "${FLAGS_RELEASE}"
        )
    endif()

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

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(benchmark PUBLIC OpenMP::OpenMP_CXX)
    endif()

endfunction()

make_benchmark("my_lapack_seq")
