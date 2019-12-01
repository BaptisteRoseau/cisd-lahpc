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

    my_dgemm_scal( CblasColMajor,
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

void benchmark_dgetf2( int n )
{
    Mat L =
        MatRandLi( n ); // Lower triangular matrix, all elements are strictly positive, and diagonal is filled with one
    Mat U = MatRandUi( n ); // Upper triangular matrix, all elements are strictly positive

    Mat LU( L.numRow(), L.numCol() );
    for ( int i = 0; i < L.numRow(); ++i ) {
        for ( int j = 0; j < i; ++j ) {
            LU.at( i, j ) = L.at( i, j );
        }
        for ( int j = i; j < L.numCol(); ++j ) {
            LU.at( i, j ) = U.at( i, j );
        }
    }

    /*Mat Prod( L.dimX(), U.dimY(), 0.0 );

    my_dgemm_openmp( CblasColMajor,
                     CblasNoTrans,
                     CblasNoTrans,
                     L.dimX(),
                     U.dimY(),
                     L.dimY(),
                     1.0,
                     L.get(),
                     L.dimX(),
                     U.get(),
                     U.dimX(),
                     0.0,
                     Prod.get(),
                     Prod.dimX() );*/

    my_dgetrf_openmp( CblasColMajor, LU.dimX(), LU.dimY(), LU.get(), LU.dimX() );
}

int main( int argc, char **argv )
{
    if ( argc != 2 ) return -1;

    int n = std::atoi( argv[1] );

    // benchmark_dgemm_scalaire( n );

    benchmark_dgetf2( n );
}
