cmake_minimum_required(VERSION 3.13.1)
project( cisd-lahpc VERSION 0.1
               DESCRIPTION "Matrix product and sequential LU factorisation"
               LANGUAGES CXX )

set_property( GLOBAL PROPERTY USE_FOLDERS ON )

add_subdirectory(lib)
add_subdirectory(perf)
add_subdirectory(test)

