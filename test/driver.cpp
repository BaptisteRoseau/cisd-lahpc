#include "Mat.h"
#include "my_lapack.h"
#include "util.h"

#include <iostream>

template<typename T, size_t N>
char ( &ArraySizeHelper( T ( & )[N] ) )[N];

#define ARRAY_SIZE( A ) ( sizeof( ArraySizeHelper( A ) ) )

using namespace my_lapack;

void test_dgetf2()
{
    Mat L = MatRandLi( 5 );
    Mat U = MatRandUi( 5 );

    L.print();
    U.print();
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

    Prod.print();
    
    int info;
    my_dgetf2( Prod.dimX(), Prod.dimY(), Prod.get(), Prod.dimX(), nullptr, &info );

    Prod.print();
}

int test_dgemm_scalaire(){
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

int test_dgemm(){
    Mat m1 = Mat( 10, 10, 1 ), m2 = Mat( 10, 10, 1 );
    Mat m5 = Mat( 10, 10, 0 );
    my_dgemm( CblasColMajor,
                       CblasTrans,
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

int main( int argc, char **argv )
{
    /*Mat m1 = MatRandi( 10, 10, 100 ), m2 = MatRandi( 10, 10, 100 );
    Mat m3 = MatSqrDiag( 50, 1 ), m4 = MatSqrDiag( 50, 999999999.654616165414169 );
    Mat m5 = Mat( 10, 10, 0 );

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
              1 ); */

    //test_dgetf2();
    test_dgemm_scalaire();
    test_dgemm();
}
