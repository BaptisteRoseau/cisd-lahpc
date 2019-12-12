#!/usr/bin/bash

buildType=Debug

if [ $# -le 0 ]; then
    echo "\n[INSTALL] No argument provided, default is Debug.\n"
elif [ ! $1 = "Debug" -a ! $1 = "Release" ]; then
    echo "\n[INSTALL] Incorrect configuration. Possible values are Debug and Release.\n";
    exit 0
else
    buildType=$1;
    echo "\n[INSTALL] Build type is $buildType\n"
fi

currDir=$(pwd)

module load compiler/gcc/8.3.0 linalg/mkl/2019_update4 compiler/cuda/10.1 mpi/openmpi/4.0.1

mkdir -p build && cd build && \
cmake3 -DCMAKE_BUILD_TYPE=$buildType -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON .. && make -j20 

if ! [ $? -eq 0 ]; then
    echo "Please run this script from the project root directory using 'source'."
    rm -r build
    cd $currDir
    exit 1
fi

cd $currDir