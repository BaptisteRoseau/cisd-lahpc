#!/usr/bin/env sh

# Make sure MKL is available
module load linalg/mkl/2019_update4

# Add the lagonum directory to the environment
export ALGONUM_DIR=~cisd-faverge/algonum/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ALGONUM_DIR/lib
export LD_RUN_PATH=$LD_RUN_PATH:$ALGONUM_DIR/lib
export INCLUDE_PATH=$INCLUDE_PATH:$ALGONUM_DIR/include

