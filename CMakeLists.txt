cmake_minimum_required(VERSION 3.10)
project( cisd-lahpc
               DESCRIPTION "Matrix product and sequential LU factorisation"
               LANGUAGES CXX C)

set_property( GLOBAL PROPERTY USE_FOLDERS ON )

enable_testing()

add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tp4)
