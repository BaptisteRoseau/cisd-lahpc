# Bulding source
mkdir -p build
cd build
cmake .. && make -j4
cd ..

args(){
    ARG=$1
    FUNC=$2
    if [ $FUNC = "dgemm" ]; then
        echo "-M $ARG -N $ARG -K $ARG";
    fi
    if [ $FUNC = "dgetrf" ]; then
        echo "-M $ARG -N $ARG";
    fi
}

# Oui, cette fonction est dégueu car j'ai voulu factoriser, non ce n'était pas une bonne idée.
# Main FUNCTION: should be called for "dgemm", "dgetrf" etc..
benchmark(){
    FUNCTION=$1
    BENCH_FILE_TIME=build/$FUNCTION\_time.txt
    BENCH_FILE_FLOPS=build/$FUNCTION\_flops.txt
    SIZES='100 500 1000' #5000 10000 50000'

    echo "$FUNCTION execution time,Size,Time(us),linear,linear"   > $BENCH_FILE_TIME
    echo "$FUNCTION execution GFlop/s,Size,GFlop/s,linear,linear" > $BENCH_FILE_FLOPS
    echo ""
    echo "$FUNCTION benchmark wrote into:"
    echo "   $(pwd)/$BENCH_FILE_TIME"
    echo "   $(pwd)/$BENCH_FILE_FLOPS"

    echo "Sequential"  >> $BENCH_FILE_TIME
    echo "Sequential"  >> $BENCH_FILE_FLOPS
    echo "Executing benchmark for $FUNCTION seq...";
    for size in $SIZES; do \
        echo "Size $size..." ; \
        start=`echo $(($(date +%s%N)/1000))`; \
        echo "$size, $(build/tp4/testings/perf_$FUNCTION -v mkl $(args $size $FUNCTION) \
        | tail -n 1 | awk -F " " '{print $9}')" 1>> $BENCH_FILE_FLOPS; \
        end=`echo $(($(date +%s%N)/1000))`; \
        echo "$size, $((end-start))" >> $BENCH_FILE_TIME; \
    done

    echo "OpenMP"  >> $BENCH_FILE_TIME
    echo "OpenMP"  >> $BENCH_FILE_FLOPS
    echo "Executing benchmark for $FUNCTION omp...";
    for size in $SIZES; do \
        echo "Size $size..." ; \
        start=`echo $(($(date +%s%N)/1000))`; \
        echo "$size, $(build/tp4/testings/perf_$FUNCTION -v mkl $(args $size $FUNCTION) \
        | tail -n 1 | awk -F " " '{print $9}')" 1>> $BENCH_FILE_FLOPS; \
        end=`echo $(($(date +%s%N)/1000))`; \
        echo "$size, $((end-start))" >> $BENCH_FILE_TIME; \
    done

    echo "MKL"  >> $BENCH_FILE_TIME
    echo "MKL"  >> $BENCH_FILE_FLOPS
    echo "Executing benchmark for $FUNCTION mkl...";
    for size in $SIZES; do \
        echo "Size $size..." ; \
        start=`echo $(($(date +%s%N)/1000))`; \
        echo "$size, $(build/tp4/testings/perf_$FUNCTION -v mkl $(args $size $FUNCTION) \
        | tail -n 1 | awk -F " " '{print $9}')" 1>> $BENCH_FILE_FLOPS; \
        end=`echo $(($(date +%s%N)/1000))`; \
        echo "$size, $((end-start))" >> $BENCH_FILE_TIME; \
    done
}

draw (){
    FUNCTION=$1
    BENCH_FILE_TIME=build/$FUNCTION\_time.txt
    BENCH_FILE_FLOPS=build/$FUNCTION\_flops.txt

    python3 scripts/drawCurve.py $BENCH_FILE_TIME
    python3 scripts/drawCurve.py $BENCH_FILE_FLOPS
}

benchmark "dgemm"
benchmark "dgetrf"

draw "dgemm"
draw "dgetrf"

#TODO: flag de compil' omp pour mkl
