currDir=$(pwd)

TEST_PERF_FILE=build/test/test_perf_my_lapack_all

if [ ! -f $TEST_PERF_FILE ]; then
    scripts/install.sh
    if ! [ $? -eq 0 ]; then
        module load language/python/3.5.9
        source scripts/install_plafrim.sh
        if ! [ $? -eq 0 ]; then
        echo "Failed to build binaries."
        cd $currDir
        exit 1
        fi
    fi
fi


cd $currDir
$TEST_PERF_FILE build/dgemm_time.csv build/dgemm_flops.csv
python3 scripts/drawCurve.py build/dgemm_time.csv
python3 scripts/drawCurve.py build/dgemm_flops.csv

if ! [ $? -eq 0 ]; then
    echo "Please run this script from the project root directory using 'source'."
fi
