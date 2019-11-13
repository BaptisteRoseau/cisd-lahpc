#include "my_lapack.h"

#include "err.h"

#include <cstdint>

namespace my_lapack {

    double my_ddot( const int N, const double *X, const int incX, const double *Y, const int incY )
    {
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( incX );
        LAHPC_CHECK_POSITIVE( incY );

        double ret = 0;
        for ( uint32_t i = 0, xi = 0, yi = 0; i < N; ++i, xi += incX, yi += incY ) { ret += X[xi] * Y[yi]; }
        return ret;
    }

    void my_daxpy( const int N, const double alpha, const double *X, const int incX, double *Y, const int incY )
    {
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( incX );
        LAHPC_CHECK_POSITIVE( incY );

        if ( alpha == 0.0 ) return;

        for ( uint32_t i = 0, xi = 0, yi = 0; i < N; ++i, xi += incX, yi += incY ) { Y[yi] += alpha * X[xi]; }
    }

    void my_dgemv( LAHPC_LAYOUT    layout,
                   LAHPC_TRANSPOSE TransA,
                   int             M,
                   int             N,
                   double          alpha,
                   const double *  A,
                   int             lda,
                   const double *  X,
                   int             incX,
                   double          beta,
                   double *        Y,
                   const int       incY )
    {
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE_STRICT( incX );
        LAHPC_CHECK_POSITIVE_STRICT( incY );
        LAHPC_CHECK_PREDICATE( layout != LAHPC_LAYOUT::RowMajor );

        if ( M == 0 || N == 0 || ( alpha == 0.0 && beta == 1.0 ) ) return;

        if ( beta != 1.0 ) {
            int lenY = ( TransA == LAHPC_TRANSPOSE::NoTrans ) ? M : N;

            if ( beta == 0 && incY == 1 ) { memset( Y, 0, lenY * sizeof( double ) ); }
            else if ( beta == 0 ) {
                for ( int i = 0, yi = 0; i < lenY; ++i, yi += incY ) { Y[yi] = 0; }
            }
            else {
                for ( int i = 0, yi = 0; i < lenY; ++i, yi += incY ) { Y[yi] *= beta; }
            }
        }

        if ( TransA == LAHPC_TRANSPOSE::NoTrans ) {
            for ( int j = 0, xi = 0; j < N; ++j, xi += incX ) {
                double tmp = alpha * X[xi];
                for ( int i = 0, yi = 0; i < M; ++i, yi += incY ) { Y[yi] += tmp * A[j * M + i]; }
            }
        }

        else if ( TransA == LAHPC_TRANSPOSE::Trans ) {
            for ( int j = 0, yi = 0; j < N; ++j, yi += incY ) {
                double tmp = 0;
                for ( int i = 0, xi = 0; i < N; ++i, xi += incX ) { tmp += A[j * M + i] * X[xi]; }
                Y[yi] += alpha * tmp;
            }
        }
    }

} // namespace my_lapack