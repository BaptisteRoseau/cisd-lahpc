#include "Mat.h"
#include "my_lapack.h"
#include "util.h"

#include <chrono>
#include <iostream>

template<typename T, size_t N>
char ( &ArraySizeHelper( T ( & )[N] ) )[N];

#define ARRAY_SIZE( A ) ( sizeof( ArraySizeHelper( A ) ) )

using namespace my_lapack;
using namespace std;

int test_perf_dgemm(){
    size_t powInc = 2;
    size_t len = 2;
    size_t lenMax = powInc << 10;
    double valMax = 200;
    double alpha  = 0.75;
    double beta   = 0;
    Mat *m1, *m2, *m3;

    while (len < lenMax)
    {
        m1 = new Mat(len, len, 10);
        m2 = new Mat(len, len, 10);
        m3 = new Mat(len, len, 10);
        auto t0 = chrono::high_resolution_clock::now();
        my_dgemm( CblasColMajor,
                  CblasNoTrans,
                  CblasNoTrans,
                  m1->dimX(),
                  m2->dimY(),
                  m1->dimY(),
                  alpha,
                  m1->get(),
                  m1->dimX(),
                  m2->get(),
                  m2->dimX(),
                  beta,
                  m3->get(),
                  m3->dimX() );
        auto t1 = chrono::high_resolution_clock::now();

        cout << "Len: " << len << "\tTime: " << (t1-t0).count() << endl; 

        delete m1;
        delete m2;
        delete m3;
        len *= powInc;
    }

    return 0;
}

int main( int argc, char **argv )
{
    test_perf_dgemm();
}
