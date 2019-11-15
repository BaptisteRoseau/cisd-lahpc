#include "my_lapack.h"

#include "err.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

static const int BLOCK_SIZE = 128;

#define AT_RM( i, j, width ) ( ( i ) * ( width ) + ( j ) )
#define AT( i, j, heigth ) ( ( j ) * ( heigth ) + ( i ) )
#define min( a, b ) ( ( a ) < ( b ) ? ( a ) : ( b ) )

namespace my_lapack {

    double my_ddot( const int N, const double *X, const int incX, const double *Y, const int incY )
    {
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( incX );
        LAHPC_CHECK_POSITIVE( incY );

        double ret = 0;
        for ( int i = 0, xi = 0, yi = 0; i < N; ++i, xi += incX, yi += incY ) { ret += X[xi] * Y[yi]; }
        return ret;
    }

    void my_daxpy( const int N, const double alpha, const double *X, const int incX, double *Y, const int incY )
    {
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( incX );
        LAHPC_CHECK_POSITIVE( incY );

        if (alpha == 0.0) { return; }

        for ( int i = 0, xi = 0, yi = 0; i < N; ++i, xi += incX, yi += incY ) { Y[yi] += alpha * X[xi]; }
    }

    void my_dgemv( enum CBLAS_LAYOUT    layout,
                   enum CBLAS_TRANSPOSE TransA,
                   int                  M,
                   int                  N,
                   double               alpha,
                   const double *       A,
                   int                  lda,
                   const double *       X,
                   int                  incX,
                   double               beta,
                   double *             Y,
                   const int            incY )
    {
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE_STRICT( incX );
        LAHPC_CHECK_POSITIVE_STRICT( incY );
        LAHPC_CHECK_PREDICATE( layout == CBLAS_LAYOUT::CblasColMajor );

        if ( M == 0 || N == 0 || ( alpha == 0.0 && beta == 1.0 ) ) return;

        if ( beta != 1.0 ) {
            int lenY = ( TransA == CBLAS_TRANSPOSE::CblasNoTrans ) ? M : N;

            if ( beta == 0 && incY == 1 ) { memset( Y, 0, lenY * sizeof( double ) ); }
            else if ( beta == 0 ) {
                for ( int i = 0, yi = 0; i < lenY; ++i, yi += incY ) { Y[yi] = 0; }
            }
            else {
                for ( int i = 0, yi = 0; i < lenY; ++i, yi += incY ) { Y[yi] *= beta; }
            }
        }

        if ( TransA == CBLAS_TRANSPOSE::CblasNoTrans ) {
            for ( int j = 0, xi = 0; j < N; ++j, xi += incX ) {
                double tmp = alpha * X[xi];
                for ( int i = 0, yi = 0; i < M; ++i, yi += incY ) { Y[yi] += tmp * A[j * M + i]; }
            }
        }

        else if ( TransA == CBLAS_TRANSPOSE::CblasTrans ) {
            for ( int j = 0, yi = 0; j < N; ++j, yi += incY ) {
                double tmp = 0;
                for ( int i = 0, xi = 0; i < N; ++i, xi += incX ) { tmp += A[j * M + i] * X[xi]; }
                Y[yi] += alpha * tmp;
            }
        }
    }

    /// M N and K aren't changed even if transposed.
    void my_dgemm_scalaire( enum CBLAS_LAYOUT    Order,
                            enum CBLAS_TRANSPOSE TransA,
                            enum CBLAS_TRANSPOSE TransB,
                            int                  M,
                            int                  N,
                            int                  K,
                            double               alpha,
                            const double *       A,
                            int                  lda,
                            const double *       B,
                            int                  ldb,
                            double               beta,
                            double *             C,
                            int                  ldc )
    {
        (void)Order; // Avoid warning
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( K );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( ldb );
        LAHPC_CHECK_POSITIVE_STRICT( ldc );

        // Avoiding useless calculus
        if ( alpha == 0. ) {
            if ( beta != 1. ) {
                size_t t;
                size_t m = M * N;
                for ( t = 0; t < m; t++ ) { C[t] *= beta; }
            }
            return;
        }

        //
        const bool bTransA = ( TransA == CBLAS_TRANSPOSE::CblasTrans );
        const bool bTransB = ( TransB == CBLAS_TRANSPOSE::CblasTrans );

        // Calculating dgemm
        size_t i, j, k; // TODO: Changer i,j,k en m,n,k
        if ( bTransA && bTransB ) {
            // Verifying dimensions validity
            if ( M != N ) {
                fprintf( stderr, "Warning: Invalid dimensions of A and B.\n" ); // TODO: More helpful message
                return;
            }

            // Computing gemm
            for ( j = 0; j < K; j++ ) {
                for ( i = 0; i < K; i++ ) {
                    for ( k = 0; k < M; k++ ) {
                        C[AT( i, j, ldc )] +=
                            alpha * A[AT( j, i, lda )] * B[AT( j, i, ldb )] + beta * C[AT( i, j, ldc )];
                    }
                }
            }
        }
        else if ( !bTransA && bTransB ) {
            // Verifying dimensions validity
            if ( N != K ) {
                fprintf( stderr, "Warning: Invalid dimensions of A and B.\n" ); // TODO: More helpful message
                return;
            }

            for ( j = 0; j < K; j++ ) {
                for ( i = 0; i < M; i++ ) {
                    for ( k = 0; k < K; k++ ) {
                        C[AT( i, j, ldc )] +=
                            alpha * A[AT( i, j, lda )] * B[AT( j, i, ldb )] + beta * C[AT( i, j, ldc )];
                    }
                }
            }
        }
        else if ( bTransA && !bTransB ) {
            // Verifying dimensions validity
            if ( M != K ) {
                fprintf( stderr, "Warning: Invalid dimensions of A and B.\n" ); // TODO: More helpful message
                return;
            }

            for ( j = 0; j < N; j++ ) {
                for ( i = 0; i < K; i++ ) {
                    for ( k = 0; k < K; k++ ) {
                        C[AT( i, j, ldc )] +=
                            alpha * A[AT( j, i, lda )] * B[AT( i, j, ldb )] + beta * C[AT( i, j, ldc )];
                    }
                }
            }
        }
        else {
            for ( j = 0; j < N; j++ ) {
                for ( i = 0; i < M; i++ ) {
                    for ( k = 0; k < K; k++ ) {
                        C[AT( i, j, ldc )] +=
                            alpha * A[AT( i, j, lda )] * B[AT( i, j, ldb )] + beta * C[AT( i, j, ldc )];
                    }
                }
            }
        }
    }

    void my_dgemm( enum CBLAS_LAYOUT    Order,
                   enum CBLAS_TRANSPOSE TransA,
                   enum CBLAS_TRANSPOSE TransB,
                   int                  M,
                   int                  N,
                   int                  K,
                   double               alpha,
                   const double *       A,
                   int                  lda,
                   const double *       B,
                   int                  ldb,
                   double               beta,
                   double *             C,
                   int                  ldc )
    {
        (void)Order;
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( K );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( ldb );
        LAHPC_CHECK_POSITIVE_STRICT( ldc );

        int blocksize = min( min( M, N ), BLOCK_SIZE );

        // Computing most of the blocks
        /* size_t i,j;
        for (i = 0; i < M/blocksize; i++){
            for (j = 0; j < N/blocksize; j++){
                my_dgemm_scalaire(Order, TransA, TransB,
                                  ???, ???, ???,
                                  alpha, A, lda,
                                  B, ldb, beta,
                                  C, ldc);
            }
        } */

        // Computing the rest of the blocks
    }

    void my_dger( enum CBLAS_LAYOUT layout,
                  int               M,
                  int               N,
                  double            alpha,
                  const double *    X,
                  int               incX,
                  const double *    Y,
                  int               incY,
                  double *          A,
                  int               lda )
    {
        LAHPC_CHECK_PREDICATE( layout == CBLAS_LAYOUT::CblasColMajor );
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( incX );
        LAHPC_CHECK_POSITIVE_STRICT( incY );

        if ( M == 0 || N == 0 || alpha == 0.0 ) { return; }

        for ( int j = 0, yi = 0; j < N; ++j, yi += incY ) {
            if ( Y[yi] == 0.0 ) { continue; }
            else {
                double tmp = alpha * Y[yi];
                for ( int i = 0, xi = 0; i < N; ++i, xi += incX ) { A[j * M + i] += tmp * X[xi]; }
            }
        }
    }

} // namespace my_lapack