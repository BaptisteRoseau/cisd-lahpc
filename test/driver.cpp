#include "my_lapack.h"
#include "Mat.h"
#include "util.h"

#include <iostream>

//#include <stdio.h>
//#include <stdlib.h>
//#include <cblas.h>
////#include <flops.h>
////#include <lapacke.h> //header include error
////#include <perf.h>

using namespace my_lapack;

int main(int argc, char **argv)
{
    Mat m(10, 10, 0);
    int k = 0;
    for (int i = 0; i < m.dimX(); ++i)
        for (int j = 0; j < m.dimY(); ++j)
        {
            m.at(i, j) = k++;
        }

    affiche(m.dimX(), m.dimY(), m.get(), 0, std::cout);
}