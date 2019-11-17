###
#
#  @copyright 2019-2019 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
#                       Univ. Bordeaux. All rights reserved.
#
#  @author Mathieu Faverge
#  @date 2019-11-13
#
###
cmake_minimum_required (VERSION 2.8)

# Disable in source directory build
set( CMAKE_DISABLE_IN_SOURCE_BUILD ON )

# use, i.e. don't skip the full RPATH for the build tree
set( CMAKE_SKIP_BUILD_RPATH FALSE )

# when building, don't use the install RPATH already
# (but later on when installing)
set( CMAKE_BUILD_WITH_INSTALL_RPATH FALSE )

option(BUILD_SHARED_LIBS
  "Build shared libraries" ON)

if ( NOT DEFINED ALGONUM_DIR )
  set( ALGONUM_DIR ${CMAKE_SOURCE_DIR}/../install/ )
endif()

include_directories( ${ALGONUM_DIR}/include )
link_directories( ${ALGONUM_DIR}/lib )

# Add the files of your own myblas library here
add_library( myblas
  mygemm.c
  mygetrf.c
  )

install(TARGETS myblas
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

