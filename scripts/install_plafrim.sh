currDir=$(pwd)

module load compiler/gcc/8.3.0 linalg/mkl/2019_update4 compiler/cuda/10.1 mpi/openmpi/4.0.1

mkdir -p build && cd build && \
 cmake3 .. && make -j20 

if ! [ $? -eq 0 ]; then
    echo "Please run this script from the project root directory."
    rm -r build
    cd $currDir
    exit 1
fi

cd $currDir