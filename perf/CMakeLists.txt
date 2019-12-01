if ( WIN32 )
    set( FLAGS_DEBUG /DEBUG /Od) 
    set( FLAGS_RELEASE /O2)
else()
    set (FLAGS_DEBUG -g  -fopenmp -Wall -Wextra -Wno-unused-parameter)
    set( FLAGS_RELEASE -O3  -fopenmp -Wall -Wextra -Wno-unused-parameter)
endif()

function(make_benchmark my_lapack_lib)

    set (benchmark_exe "benchmark_${my_lapack_lib}")

    add_executable(
        ${benchmark_exe}
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cpp
    )
    
    target_include_directories(
        ${benchmark_exe}
        PRIVATE
            "${PROJECT_SOURCE_DIR}/lib"
    )
    
    if ( WIN32 )
        target_compile_options( 
            ${benchmark_exe}
            PRIVATE
                "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
                "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
        )
    else()
        target_compile_options( 
            ${benchmark_exe}
            PRIVATE
                "${FLAGS_DEBUG}"
                "${FLAGS_RELEASE}"
        )
    endif()
    
    set_target_properties(
        ${benchmark_exe}
        PROPERTIES CXX_STANDARD 11
    )

    target_link_libraries(
        ${benchmark_exe}
        PRIVATE 
            ${my_lapack_lib}
    )

endfunction()

#make_benchmark("my_lapack_seq")
make_benchmark("my_lapack_all")
