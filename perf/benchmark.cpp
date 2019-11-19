#include "Mat.h"
#include "my_lapack.h"
#include "util.h"

#include <cstdlib>
#include <iostream>
#include <limits>

using namespace my_lapack;

void benchmark_dgemm_scalaire( int n )
{
    Mat m1 = MatRandi( n, n, std::numeric_limits<int>::max() );
    Mat m2 = MatRandi( n, n, std::numeric_limits<int>::max() );
    Mat prod( n, n, 0.0 );

    my_dgemm_scalaire( CblasColMajor,
                       CblasNoTrans,
                       CblasNoTrans,
                       m1.dimX(),
                       m2.dimY(),
                       m1.dimX(),
                       10.0,
                       m1.get(),
                       m1.dimX(),
                       m2.get(),
                       m2.dimX(),
                       0.0,
                       prod.get(),
                       prod.dimX() );
}

int main( int argc, char **argv )
{
    if ( argc != 2 ) return -1;

    int n = std::atoi( argv[1] );

    benchmark_dgemm_scalaire( n );
}
