
if [ $# -le 1 ]; then
    echo "[INSTALL] No argument provided, default is Debug."
elif [ ! $1 = "Debug" -a ! $1 = "Release" ]; then
    echo "[INSTALL] Incorrect configuration. Possible values are Debug and Release.";
    exit 0
fi

currDir=$(pwd)

mkdir -p build && cd build && \
cmake -DCMAKE_BUILD_TYPE=$1 .. && make -j8

if ! [ $? -eq 0 ]; then
    echo "Please run this script from the project root directory using 'source'."
    rm -r build
    cd $currDir
    exit 1
fi

cd $currDir