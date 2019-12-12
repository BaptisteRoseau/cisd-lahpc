currDir=$(pwd)

mkdir -p build && cd build && \
cmake .. && make -j4 && ./test/test_perf_my_lapack_all && \

cd $currDir