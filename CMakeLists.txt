cmake_minimum_required(VERSION 2.8)
project( cisd-lahpc
               DESCRIPTION "Matrix product and sequential LU factorisation"
               LANGUAGES CXX )

set_property( GLOBAL PROPERTY USE_FOLDERS ON )

add_subdirectory(lib)
add_subdirectory(perf)
add_subdirectory(test)

