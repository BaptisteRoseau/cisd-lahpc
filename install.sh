#!/usr/bin/bash

if [ $# -le 1 ]
    echo "[INSTALL] No argument provided, default is Debug"
elif [ ! $1 = "Debug" -a ! $1 = "Release" ]
    echo "[INSTALL] Incorrect configuration. Possible values are Debug and Release"
else
    rm -rf build
    mkdir build
    cd build
    cmake3 -DCMAKE_BUILD_TYPE=$1 
fi