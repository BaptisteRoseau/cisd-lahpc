cmake_minimum_required(VERSION 3.14.1)
project( cisd-lahpc VERSION 0.1
               DESCRIPTION "Matrix product and sequential LU factorisation"
               LANGUAGES CXX )

set_property( GLOBAL PROPERTY USE_FOLDERS ON )

if( WIN32 )
    add_definitions ( "-DLAHPC_WIN" )
endif()

add_subdirectory(lib)

add_subdirectory(test)
