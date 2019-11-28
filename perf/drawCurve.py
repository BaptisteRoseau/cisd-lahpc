import matplotlib.pyplot as plt
import sys
import csv

if len(sys.argv) < 2:
    print("Usage: python3 drawCurve.py <my_file1.csv> <my_file2.csv>..")
    exit(0)


for fileIdx in range(1, len(sys.argv)):
    data = []
    plt.figure(fileIdx)
    with open(sys.argv[fileIdx],'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')  
        for row in reader:
            data.append(row)

    # Getting curve information
    title  = data[0][0]
    xtitle = data[0][1]
    ytitle = data[0][2]
    xscale = data[0][3]
    yscale = data[0][4]

    # Retrieving curves information
    i = 1
    while i < len(data):
        curveTitle = data[i][0]
        i += 1
        x, y = [], []
        while i < len(data) and len(data[i]) > 1:
            x.append(float(data[i][0]))
            y.append(float(data[i][1]))    
            i += 1
        plt.plot(x,y, label=curveTitle)

    # Ploting legend
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.yscale(str(xscale).replace(" ", ""))
    plt.xscale(str(yscale).replace(" ", ""))
    plt.legend()
    plt.grid()
plt.show()