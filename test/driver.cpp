#include "Mat.h"
#include "my_lapack.h"
#include "util.h"

#include <iostream>

template<typename T, size_t N>
char ( &ArraySizeHelper( T ( & )[N] ) )[N];

#define ARRAY_SIZE( A ) ( sizeof( ArraySizeHelper( A ) ) )

using namespace my_lapack;

void test_my_ddot() {}

int main( int argc, char **argv )
{
    Mat m1 =  MatRandi( 10, 10, 100 ), m2 = MatRandi( 10, 10, 100 );
    Mat m3 = MatSqrDiag(50, 1), m4 = MatSqrDiag(50, 999999999.654616165414169);

    my_dgemv( CBLAS_ORDER::CblasColMajor,
              CBLAS_TRANSPOSE::CblasNoTrans,
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

    affiche( m4.dimX(), m4.dimY(), m4.get(), 0, std::cout, 3 );

    /* affiche(CBLAS_ORDER::CblasColMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            ); */
}