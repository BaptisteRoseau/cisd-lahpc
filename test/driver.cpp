#include "Mat.h"
#include "my_lapack.h"
#include "util.h"

#include <iostream>

template<typename T, size_t N>
char ( &ArraySizeHelper( T ( & )[N] ) )[N];

#define ARRAY_SIZE( A ) ( sizeof( ArraySizeHelper( A ) ) )

using namespace my_lapack;

int test_dgemv()
{
    Mat m1 = MatRandi( 10, 10, 100 );
    Mat m4 = MatSqrDiag( 50, 999999999.654616165414169 );
    my_dgemv( CblasColMajor,
              CblasNoTrans,
              m1.dimX(),
              m1.dimY(),
              1.0,
              m1.get(),
              0,
              m4.col( 0 ),
              1,
              1.0,
              m4.col( 1 ),
              1 );

    m4.print();
    return 0;
}

int test_dgetf2()
{
    Mat L = MatRandLi( 1000 );
    Mat U = MatRandUi( 1000 );

    /*L.print();
    U.print();*/
    Mat Prod( L.dimX(), L.dimX(), 0.0 );

    my_dgemm_scalaire( CblasColMajor,
                       CblasNoTrans,
                       CblasNoTrans,
                       L.dimX(),
                       L.dimY(),
                       L.dimX(),
                       1.0,
                       L.get(),
                       L.dimX(),
                       U.get(),
                       U.dimX(),
                       1.0,
                       Prod.get(),
                       Prod.dimX() );

    // Prod.print();
    // my_dgetf2( Prod.dimX(), Prod.dimY(), Prod.get(), Prod.dimX(), nullptr, &info );

    // Prod.print();
    return 0;
}

int test_dgemm_scalaire()
{
    Mat m1 = Mat( 2, 1, 1 ), m2 = Mat( 1, 2, 1 );
    Mat m5 = Mat( 2, 2, 0 );
    my_dgemm_scalaire( CblasColMajor,
                       CblasNoTrans,
                       CblasNoTrans,
                       m1.dimX(),
                       m2.dimY(),
                       m1.dimY(),
                       2,
                       m1.get(),
                       m1.dimX(),
                       m2.get(),
                       m2.dimX(),
                       2,
                       m5.get(),
                       m5.dimX() );
    m5.print();
    return 0;
}

int test_dgemm()
{
    Mat m1 = Mat( 100, 200, 1 ), m2 = Mat( 200, 100, 1 );
    Mat m5 = Mat( 100, 100, 0 );
    my_dgemm( CblasColMajor,
              CblasNoTrans,
              CblasNoTrans,
              m1.dimX(),
              m2.dimY(),
              m1.dimY(),
              1,
              m1.get(),
              m1.dimX(),
              m2.get(),
              m2.dimX(),
              1,
              m5.get(),
              m5.dimX() );
    m5.print();
    return 0;
}

int main( int argc, char **argv )
{
    test_dgemv();
    test_dgetf2();
    test_dgemm_scalaire();
    test_dgemm();
}
