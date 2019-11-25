## COMMON

if ( WIN32 )
    set( FLAGS_DEBUG /DEBUG /Od ) 
    set( FLAGS_RELEASE /O2 )
else()
    set (FLAGS_DEBUG -g -Wall -Wextra)
    set( FLAGS_RELEASE -O3 -Wall -Wextra)
endif()

############################################################
######################   FUNCTIONS DEFINITION   ############
############################################################

########## Compile command for executables
function(make test_filename my_lapack_lib)

    add_executable(
        "${test_filename}_${my_lapack_lib}"
        ${CMAKE_CURRENT_SOURCE_DIR}/${test_filename}.cpp
    )
    
    target_include_directories(
        "${test_filename}_${my_lapack_lib}"
        PRIVATE
        "${PROJECT_SOURCE_DIR}/lib"
    )
    
    if ( WIN32 )
        target_compile_options( 
            "${test_filename}_${my_lapack_lib}"
            PRIVATE
                "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
                "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
        )
    else()
        target_compile_options( 
            "${test_filename}_${my_lapack_lib}"
            PRIVATE
                "${FLAGS_DEBUG}"
                "${FLAGS_RELEASE}"
        )
    endif()
    
    set_target_properties(
        "${test_filename}_${my_lapack_lib}"
        PROPERTIES CXX_STANDARD 11
    )
    
    target_link_libraries(
        "${test_filename}_${my_lapack_lib}"
        PRIVATE 
            ${my_lapack_lib}
    )

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries("${test_filename}_${my_lapack_lib}" PUBLIC OpenMP::OpenMP_CXX)
    endif()

endfunction()


############################################################
######################   CALL   ############################
############################################################


make("driver" "my_lapack_seq")
make("driver" "my_lapack_omp")

make("test_valid" "my_lapack_seq")
make("test_valid" "my_lapack_omp")

make("test_perf" "my_lapack_seq")
make("test_perf" "my_lapack_omp")
