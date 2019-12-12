#!/usr/bin/bash

if [ $# -le 1 ]; then
    echo "[INSTALL] No argument provided, default is Debug."
elif [ ! $1 = "Debug" -a ! $1 = "Release" ]; then
    echo "[INSTALL] Incorrect configuration. Possible values are Debug and Release.";
    exit 0
fi

rm -rf build
mkdir build
cd build
cmake3 -DCMAKE_BUILD_TYPE=$1 ..
