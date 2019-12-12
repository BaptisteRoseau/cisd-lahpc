currDir=$(pwd)

./install.sh && \
python3 ../perf/drawCurve.py dgemm.csv

cd $currDir
