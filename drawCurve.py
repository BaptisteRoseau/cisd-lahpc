import matplotlib.pyplot as plt
import sys
import csv

if len(sys.argv) != 2:
    print("Usage: python3 drawCurve.py <my_file.csv>")
    exit(0)

x = []
y = []

with open(sys.argv[1],'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))

plt.plot(x,y)
plt.title("dgemm time taken regarding matrix dimension")
plt.xlabel("Matrix dimension")
plt.ylabel("Time (s)")
plt.yscale("log")
plt.xscale("log")
plt.grid()
plt.show()