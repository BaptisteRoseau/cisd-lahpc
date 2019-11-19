cmake_minimum_required(VERSION 3.13.1)
project( cisd-lahpc VERSION 0.1
               DESCRIPTION "Matrix product and sequential LU factorisation"
               LANGUAGES CXX )

set_property( GLOBAL PROPERTY USE_FOLDERS ON )

add_subdirectory(lib_seq)

add_subdirectory(lib_omp)

add_subdirectory(test)

