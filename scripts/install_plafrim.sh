#!/usr/bin/bash

if [ $# -le 1 ]; then
    echo "[INSTALL] No argument provided, default is Debug."
elif [ ! $1 = "Debug" -a ! $1 = "Release" ]; then
    echo "[INSTALL] Incorrect configuration. Possible values are Debug and Release.";
    exit 0
fi

currDir=$(pwd)

module load compiler/gcc/8.3.0 linalg/mkl/2019_update4 compiler/cuda/10.1 mpi/openmpi/4.0.1

mkdir -p build && cd build && \
 cmake3 -DCMAKE_BUILD_TYPE=$1 .. && make -j20 

if ! [ $? -eq 0 ]; then
    echo "Please run this script from the project root directory using 'source'."
    rm -r build
    cd $currDir
    exit 1
fi

cd $currDir