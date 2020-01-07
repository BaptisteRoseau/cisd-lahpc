# Bulding source
mkdir -p build
cd build
cmake .. && make -j4
cd ..

SIZES='100 500 1000' # 5000 10000 50000'


# BENCHMARKING DGEMM

FUNCTION="dgemm"
BENCH_FILE_TIME=build/$FUNCTION\_time.txt
BENCH_FILE_FLOPS=build/$FUNCTION\_flops.txt

rm $BENCH_FILE_TIME $BENCH_FILE_FLOPS

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
    echo "$size, $(build/tp4/testings/perf_$FUNCTION -v mkl -M $size -N $size -K $size \
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
    echo "$size, $(build/tp4/testings/perf_$FUNCTION -v mkl -M $size -N $size -K $size \
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
    echo "$size, $(build/tp4/testings/perf_$FUNCTION -v mkl -M $size -N $size -K $size \
    | tail -n 1 | awk -F " " '{print $9}')" 1>> $BENCH_FILE_FLOPS; \
    end=`echo $(($(date +%s%N)/1000))`; \
    echo "$size, $((end-start))" >> $BENCH_FILE_TIME; \
done


# BENCHMARKING DGETRF

FUNCTION="dgetrf"
BENCH_FILE_TIME=build/$FUNCTION\_time.txt
BENCH_FILE_FLOPS=build/$FUNCTION\_flops.txt

rm $BENCH_FILE_TIME $BENCH_FILE_FLOPS

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
    echo "$size, $(build/tp4/testings/perf_$FUNCTION -v mkl -M $size -N $size \
    | tail -n 1 | awk -F " " '{print $6}')" 1>> $BENCH_FILE_FLOPS; \
    end=`echo $(($(date +%s%N)/1000))`; \
    echo "$size, $((end-start))" >> $BENCH_FILE_TIME; \
done

echo "OpenMP"  >> $BENCH_FILE_TIME
echo "OpenMP"  >> $BENCH_FILE_FLOPS
echo "Executing benchmark for $FUNCTION omp...";
for size in $SIZES; do \
    echo "Size $size..." ; \
    start=`echo $(($(date +%s%N)/1000))`; \
    echo "$size, $(build/tp4/testings/perf_$FUNCTION -v mkl -M $size -N $size \
    | tail -n 1 | awk -F " " '{print $6}')" 1>> $BENCH_FILE_FLOPS; \
    end=`echo $(($(date +%s%N)/1000))`; \
    echo "$size, $((end-start))" >> $BENCH_FILE_TIME; \
done

echo "MKL"  >> $BENCH_FILE_TIME
echo "MKL"  >> $BENCH_FILE_FLOPS
echo "Executing benchmark for $FUNCTION mkl...";
for size in $SIZES; do \
    echo "Size $size..." ; \
    start=`echo $(($(date +%s%N)/1000))`; \
    echo "$size, $(build/tp4/testings/perf_$FUNCTION -v mkl -M $size -N $size \
    | tail -n 1 | awk -F " " '{print $6}')" 1>> $BENCH_FILE_FLOPS; \
    end=`echo $(($(date +%s%N)/1000))`; \
    echo "$size, $((end-start))" >> $BENCH_FILE_TIME; \
done


draw (){
    FUNCTION=$1
    BENCH_FILE_TIME=build/$FUNCTION\_time.txt
    BENCH_FILE_FLOPS=build/$FUNCTION\_flops.txt

    python3 scripts/drawCurve.py $BENCH_FILE_TIME &
    python3 scripts/drawCurve.py $BENCH_FILE_FLOPS &
}

echo "Plotting results.."
draw "dgemm"
draw "dgetrf"
