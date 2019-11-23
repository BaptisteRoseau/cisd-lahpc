## COMMON

if ( WIN32 )
    set( FLAGS_DEBUG /DEBUG /Od ) 
    set( FLAGS_RELEASE /O2 )
else()
    set (FLAGS_DEBUG -g)
    set( FLAGS_RELEASE -O3 )
endif()

############################################################
######################   FUNCTIONS DEFINITION   ############
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

########## test_valid
function(make_test_valid my_lapack_lib)

    add_executable(
        "test_valid_${my_lapack_lib}"
        ${CMAKE_CURRENT_SOURCE_DIR}/test_valid.cpp
    )
    
    target_include_directories(
        "test_valid_${my_lapack_lib}"
        PRIVATE
        "${PROJECT_SOURCE_DIR}/lib"
    )
    
    target_compile_options( 
        "test_valid_${my_lapack_lib}"
        PRIVATE
            "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
            "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
    )
    
    set_target_properties(
        "test_valid_${my_lapack_lib}"
        PROPERTIES CXX_STANDARD 11
    )
    
    target_link_libraries(
        "test_valid_${my_lapack_lib}"
        PRIVATE 
            ${my_lapack_lib}
    )

endfunction()


########## test_perf
function(make_test_perf my_lapack_lib)

    add_executable(
        "test_perf_${my_lapack_lib}"
        ${CMAKE_CURRENT_SOURCE_DIR}/test_perf.cpp
    )
    
    target_include_directories(
        "test_perf_${my_lapack_lib}"
        PRIVATE
        "${PROJECT_SOURCE_DIR}/lib"
    )
    
    target_compile_options( 
        "test_perf_${my_lapack_lib}"
        PRIVATE
            "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
            "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
    )
    
    set_target_properties(
        "test_perf_${my_lapack_lib}"
        PROPERTIES CXX_STANDARD 11
    )
    
    target_link_libraries(
        "test_perf_${my_lapack_lib}"
        PRIVATE 
            ${my_lapack_lib}
    )

endfunction()



## NOT WORKING
##function(initialize_function test_filename)
##    function(make_${test_filename} my_lapack_lib)
##
##        add_executable(
##            "${test_filename}_${my_lapack_lib}"
##            ${CMAKE_CURRENT_SOURCE_DIR}/${test_filename}.cpp
##        )
##        
##        target_include_directories(
##            "${test_filename}_${my_lapack_lib}"
##            PRIVATE
##            "${PROJECT_SOURCE_DIR}/lib"
##        )
##        
##        target_compile_options( 
##            "${test_filename}_${my_lapack_lib}"
##            PRIVATE
##                "$<$<CONFIG:Debug>:${FLAGS_DEBUG}>"
##                "$<$<CONFIG:Release>:${FLAGS_RELEASE}>"
##        )
##        
##        set_target_properties(
##            "${test_filename}_${my_lapack_lib}"
##            PROPERTIES CXX_STANDARD 11
##        )
##        
##        target_link_libraries(
##            "${test_filename}_${my_lapack_lib}"
##            PRIVATE 
##                ${my_lapack_lib}
##        )
##
##    endfunction()
##endfunction()
##
##initialize_function("test_valid")
##initialize_function("test_perf")
##
##make_test_perf("my_lapack_omp")


############################################################
######################   CALL   ############################
############################################################


make_driver("my_lapack_seq")
make_driver("my_lapack_omp")


make_test_valid("my_lapack_seq")
make_test_valid("my_lapack_omp")

make_test_perf("my_lapack_seq")
make_test_perf("my_lapack_omp")
