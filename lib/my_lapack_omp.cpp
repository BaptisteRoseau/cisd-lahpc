#include "err.h"
#include "my_lapack.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <omp.h>
#include <utility>

#define _LAHPC_BLOCK_SIZE 128
static const int BLOCK_SIZE = _LAHPC_BLOCK_SIZE;

#define AT_RM( i, j, width ) ( ( i ) * ( width ) + ( j ) )
#define AT( i, j, heigth ) ( ( j ) * ( heigth ) + ( i ) )
#define min_macro( a, b ) ( ( a ) < ( b ) ? ( a ) : ( b ) )

//TODO: simd if incX == 1

namespace my_lapack {

    double my_ddot( const int N, const double *X, const int incX, const double *Y, const int incY )
    {
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( incX );
        LAHPC_CHECK_POSITIVE( incY );

        double ret = 0;
        //TODO: reduction
        for ( int i = 0, xi = 0, yi = 0; i < N; ++i, xi += incX, yi += incY ) { ret += X[xi] * Y[yi]; }
        return ret;
    }

    void my_daxpy( const int N, const double alpha, const double *X, const int incX, double *Y, const int incY )
    {
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( incX );
        LAHPC_CHECK_POSITIVE( incY );

        if ( alpha == 0.0 ) { return; }

        #pragma omp parallel default(shared)
        {
            int xi, yi;
            #pragma omp for
            for ( int i = 0; i < N; i++) { 
                yi = i*incY;
                xi = i*incX;
                Y[yi] += alpha * X[xi];
            }

        }
    }

    void my_dgemv( CBLAS_ORDER     layout,
                   CBLAS_TRANSPOSE TransA,
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
        LAHPC_CHECK_PREDICATE( layout == CblasColMajor );

        if ( M == 0 || N == 0 || ( alpha == 0.0 && beta == 1.0 ) ) return;

        if ( beta != 1.0 ) {
            int lenY = ( TransA == CblasNoTrans ) ? M : N;

            if ( beta == 0 && incY == 1 ) { memset( Y, 0, lenY * sizeof( double ) ); }
            else if ( beta == 0 ) {
                int len = lenY*incY;
                #pragma omp parallel for default(shared)
                for ( int yi = 0; yi < len; yi += incY ) { Y[yi] = 0; }
            }
            else {
                int len = lenY*incY;
                #pragma omp parallel for default(shared)
                for ( int yi = 0; yi < len; yi += incY ) { Y[yi] *= beta; }
            }
        }

        if ( TransA == CblasNoTrans ) {
            #pragma omp parallel for default(shared)
            for ( int j = 0; j < N; ++j) {
                double tmp = alpha * X[incX*j];
                for ( int i = 0; i < M; i++) { Y[i*incY] += tmp * A[j * lda + i]; }
            }
        }

        else if ( TransA == CblasTrans ) {
            #pragma omp parallel for default(shared)
            for ( int j = 0; j < N; ++j) {
                int yi = j*incY;
                double tmp = 0;
                for ( int i = 0, xi = 0; i < M; ++i, xi += incX ) { tmp += A[j * lda + i] * X[xi]; }
                Y[yi] += alpha * tmp;
            }
        }
    }

    //TODO: reduce on linear add
    void my_dgemm_scalaire( CBLAS_ORDER     Order,
                            CBLAS_TRANSPOSE TransA,
                            CBLAS_TRANSPOSE TransB,
                            int             M,
                            int             N,
                            int             K,
                            double          alpha,
                            const double *  A,
                            int             lda,
                            const double *  B,
                            int             ldb,
                            double          beta,
                            double *        C,
                            int             ldc )
    {
        LAHPC_CHECK_PREDICATE( Order == CblasColMajor );
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( K );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( ldb );
        LAHPC_CHECK_POSITIVE_STRICT( ldc );

        // Early return
        if ( !M || !N || !K ) { return; }

        // Early return
        if ( alpha == 0. ) {
            if ( beta != 1. ) {
                int m, n;
#pragma omp parallel for collapse( 2 )
                for ( m = 0; m < M; m++ ) {
                    for ( n = 0; n < N; n++ ) { C[AT( m, n, ldc )] *= beta; }
                }
            }
            return;
        }

        // Computing booleans in advance
        bool bTransA = ( TransA == CblasTrans );
        bool bTransB = ( TransB == CblasTrans );

        // Calculating dgemm
        int m, n, k;
        if ( bTransA && bTransB ) {
#pragma omp parallel for default( shared ) collapse( 2 ) private( k )
            for ( n = 0; n < N; n++ ) {
                for ( m = 0; m < M; m++ ) {
                    C[AT( m, n, ldc )] *= beta;
                    for ( k = 0; k < K; k++ ) { C[AT( m, n, ldc )] += alpha * A[AT( n, k, lda )] * B[AT( k, m, ldb )]; }
                }
            }
        }
        else if ( !bTransA && bTransB ) {
#pragma omp parallel for default( shared ) collapse( 2 ) private( k )
            for ( n = 0; n < N; n++ ) {
                for ( m = 0; m < M; m++ ) {
                    C[AT( m, n, ldc )] *= beta;
                    for ( k = 0; k < K; k++ ) { C[AT( m, n, ldc )] += alpha * A[AT( m, k, lda )] * B[AT( k, m, ldb )]; }
                }
            }
        }
        else if ( bTransA && !bTransB ) {
#pragma omp parallel for default( shared ) collapse( 2 ) private( k )

            for ( n = 0; n < N; n++ ) {
                for ( m = 0; m < M; m++ ) {
                    C[AT( m, n, ldc )] *= beta;
                    for ( k = 0; k < K; k++ ) { C[AT( m, n, ldc )] += alpha * A[AT( n, k, lda )] * B[AT( k, n, ldb )]; }
                }
            }
        }
        else {
#pragma omp parallel for default( shared ) collapse( 2 ) private( k )
            for ( n = 0; n < N; n++ ) {
                for ( m = 0; m < M; m++ ) {
                    C[AT( m, n, ldc )] *= beta;
                    for ( k = 0; k < K; k++ ) { C[AT( m, n, ldc )] += alpha * A[AT( m, k, lda )] * B[AT( k, n, ldb )]; }
                }
            }
        }
    }

    void my_dgemm( CBLAS_ORDER     Order,
                   CBLAS_TRANSPOSE TransA,
                   CBLAS_TRANSPOSE TransB,
                   int             M,
                   int             N,
                   int             K,
                   double          alpha,
                   const double *  A,
                   int             lda,
                   const double *  B,
                   int             ldb,
                   double          beta,
                   double *        C,
                   int             ldc )
    {
        LAHPC_CHECK_PREDICATE( Order == CblasColMajor );
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( K );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( ldb );
        LAHPC_CHECK_POSITIVE_STRICT( ldc );

        // Early return
        if ( !M || !N || !K || ( alpha == 0. && beta == 1. ) ) { return; }
        if ( alpha == 0 ) {
            my_dgemm_scalaire( Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc );
            return;
        }

        // Computing booleans in advance
        bool bTransA = ( TransA == CblasTrans );
        bool bTransB = ( TransB == CblasTrans );

        int blocksize = min_macro( min_macro( M, N ), BLOCK_SIZE );
        int lastMB = M % blocksize, MB = M / blocksize + 1;
        int lastNB = N % blocksize, NB = N / blocksize + 1;
        int lastKB = K % blocksize, KB = K / blocksize + 1;
        int m, n, k;
        if ( bTransA && bTransB ) {
#pragma omp parallel for default( shared ) collapse( 2 ) private( k )
            for ( m = 0; m < MB; m++ ) {
                for ( n = 0; n < NB; n++ ) {
                    for ( k = 0; k < KB; k++ ) {
                        my_dgemm_scalaire( Order,
                                           TransA,
                                           TransB,
                                           m < MB - 1 ? blocksize : lastMB,
                                           n < NB - 1 ? blocksize : lastNB,
                                           k < KB - 1 ? blocksize : lastKB,
                                           alpha,
                                           A + blocksize * AT( k, m, lda ),
                                           lda,
                                           B + blocksize * AT( n, k, ldb ),
                                           ldb,
                                           beta,
                                           C + blocksize * AT( m, n, ldc ),
                                           ldc );
                    }
                }
            }
        }
        else if ( !bTransA && bTransB ) {
#pragma omp parallel for default( shared ) collapse( 2 ) private( k )
            for ( m = 0; m < MB; m++ ) {
                for ( n = 0; n < NB; n++ ) {
                    for ( k = 0; k < KB; k++ ) {
                        my_dgemm_scalaire( Order,
                                           TransA,
                                           TransB,
                                           m < MB - 1 ? blocksize : lastMB,
                                           n < NB - 1 ? blocksize : lastNB,
                                           k < KB - 1 ? blocksize : lastKB,
                                           alpha,
                                           A + blocksize * AT( m, k, lda ),
                                           lda,
                                           B + blocksize * AT( n, k, ldb ),
                                           ldb,
                                           beta,
                                           C + blocksize * AT( m, n, ldc ),
                                           ldc );
                    }
                }
            }
        }
        else if ( bTransA && !bTransB ) {
#pragma omp parallel for default( shared ) collapse( 2 ) private( k )
            for ( m = 0; m < MB; m++ ) {
                for ( n = 0; n < NB; n++ ) {
                    for ( k = 0; k < KB; k++ ) {
                        my_dgemm_scalaire( Order,
                                           TransA,
                                           TransB,
                                           m < MB - 1 ? blocksize : lastMB,
                                           n < NB - 1 ? blocksize : lastNB,
                                           k < KB - 1 ? blocksize : lastKB,
                                           alpha,
                                           A + blocksize * AT( k, m, lda ),
                                           lda,
                                           B + blocksize * AT( k, n, ldb ),
                                           ldb,
                                           beta,
                                           C + blocksize * AT( m, n, ldc ),
                                           ldc );
                    }
                }
            }
        }
        else {
#pragma omp parallel for default( shared ) collapse( 2 ) private( k )
            for ( m = 0; m < MB; m++ ) {
                for ( n = 0; n < NB; n++ ) {
                    for ( k = 0; k < KB; k++ ) {
                        my_dgemm_scalaire( Order,
                                           TransA,
                                           TransB,
                                           m < MB - 1 ? blocksize : lastMB,
                                           n < NB - 1 ? blocksize : lastNB,
                                           k < KB - 1 ? blocksize : lastKB,
                                           alpha,
                                           A + blocksize * AT( m, k, lda ),
                                           lda,
                                           B + blocksize * AT( k, n, ldb ),
                                           ldb,
                                           beta,
                                           C + blocksize * AT( m, n, ldc ),
                                           ldc );
                    }
                }
            }
        }
    }

    void my_dger( CBLAS_ORDER   layout,
                  int           M,
                  int           N,
                  double        alpha,
                  const double *X,
                  int           incX,
                  const double *Y,
                  int           incY,
                  double *      A,
                  int           lda )
    {
        LAHPC_CHECK_PREDICATE( layout == CblasColMajor );
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( incX );
        LAHPC_CHECK_POSITIVE_STRICT( incY );

        if ( M == 0 || N == 0 || alpha == 0.0 ) { return; }

        #pragma omp parallel for default(shared)
        for ( int j = 0; j < N; ++j) {
            int yi = j*incY;
            if ( Y[yi] == 0.0 ) { continue; }
            else {
                double tmp = alpha * Y[yi];
                #pragma omp parallel for
                for ( int i = 0; i < N; ++i) { A[j * lda + i] += tmp * X[i*incX]; }
            }
        }
    }

    void my_dgetf2( CBLAS_ORDER order, int M, int N, double *A, int lda )
    {
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_PREDICATE( lda >= std::max( 1, M ) );

        if ( M == 0 || N == 0 ) { return; }

        int minMN = std::min( M, N );

        #pragma omp parallel for
        for ( int j = 0; j < minMN; ++j ) {
            if ( j < M - 1 ) {
                if ( std::abs( A[j * lda + j] ) > ( 2.0 * std::numeric_limits<double>::epsilon() ) ) {
                    my_dscal( M - j - 1, 1.0 / A[j * lda + j], A + j * lda + j + 1, 1 );
                }
                else {
                    for ( int i = 0; i < M - j; ++i ) { A[j * lda + j + i] /= A[j * lda + j]; }
                }
            }
            if ( j < minMN - 1 ) {
                my_dger( CblasColMajor,
                         M - j - 1,
                         N - j - 1,
                         -1.0,
                         A + j * lda + j + 1,
                         1,
                         A + ( j + 1 ) * lda + j,
                         lda,
                         A + ( j + 1 ) * lda + j + 1,
                         lda );
            }
        }
    }

    //TODO
    void my_dgetrf( CBLAS_ORDER order, int M, int N, double *A, int lda ) {}

    void my_dtrsm( CBLAS_ORDER     layout,
                   CBLAS_SIDE      side,
                   CBLAS_UPLO      uplo,
                   CBLAS_TRANSPOSE transA,
                   CBLAS_DIAG      diag,
                   int             M,
                   int             N,
                   double          alpha,
                   const double *  A,
                   int             lda,
                   double *        B,
                   int             ldb )
    {
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( lda );
        LAHPC_CHECK_POSITIVE( ldb );

        LAHPC_CHECK_PREDICATE( side == CblasLeft );

        double lambda;

        if ( M == 0 || N == 0 ) return;

        if ( alpha == 0. ) {
            #pragma omp parallel for simd collapse(2)
            for ( int j = 0; j < N; ++j ) {
                for ( int i = 0; i < M; ++i ) { B[i + j * ldb] = 0.; }
            }
            return;
        }

        /* Left side : X * op( A ) = alpha * B */
        if ( side == CblasLeft ) {
            /* B = alpha * inv(A ** t) * B */
            if ( transA == CblasTrans ) {
                /* A is a lower triangular */
                if ( uplo == CblasLower ) {
                    for ( int j = 0; j < N; ++j ) {
                        for ( int i = M - 1; i >= 0; --i ) {
                            lambda = alpha * B[i + j * ldb];
                            for ( int k = i + 1; k < M; ++k ) { lambda -= B[k + j * ldb] * A[k + i * lda]; }
                            /* The diagonal is A[i + i*lda] (Otherwise : 1.) */
                            /* Relevent when solving A = L*U as we use A to store
                             both L and U, so Diag(L) is full of 1. . */
                            if ( diag == CblasNonUnit ) lambda /= A[i * ( 1 + lda )];
                            B[i + j * ldb] = lambda;
                        }
                    }
                }
                /* A is triangular upper */
                else if ( uplo == CblasUpper ) {
                    for ( int j = 0; j < N; ++j ) {
                        for ( int i = 0; i < M; ++i ) {
                            lambda = alpha * B[i + j * ldb];
                            for ( int k = 0; k < i; ++k ) { lambda -= A[k + i * lda] * B[k + j * ldb]; }
                            if ( diag == CblasNonUnit ) lambda /= A[i * ( 1 + lda )];
                            B[i + j * ldb] = lambda;
                        }
                    }
                }
            }
        }
        else {
            if ( transA == CblasTrans ) {
                if ( uplo == CblasUpper ) {}
                else {
                }
            }
            else {
                if ( uplo == CblasUpper ) {}
                else {
                }
            }
        }
    }
    int my_idamax( int N, double *dx, int incX )
    {
        LAHPC_CHECK_POSITIVE_STRICT( N );
        LAHPC_CHECK_POSITIVE_STRICT( incX );

        int idamax = -1;

        if ( N <= 0 ) { return idamax; }

        idamax     = 0;
        double max = std::abs( dx[0] );
        for ( int i = 1, xi = incX; i < N; ++i, xi += incX ) {
            double tmp = std::abs( dx[i] );
            if ( tmp > max ) {
                idamax = i;
                max    = tmp;
            }
        }

        return idamax;
    }

    void my_dscal( int N, double da, double *dx, int incX )
    {
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE_STRICT( incX );

        if ( N == 0 ) { return; }
        if ( da == 0.0 && incX == 1 ) {
            std::memset( dx, 0, N * sizeof( double ) );
            return;
        }
        if ( da == 0.0 ) {
            for ( int i = 0, xi = 0; i < N; ++i, xi += incX ) { dx[xi] = 0.0; }
        }
        else {
            for ( int i = 0, xi = 0; i < N; ++i, xi += incX ) { dx[xi] *= da; }
        }
    }

    void my_dlaswp( int N, double *A, int lda, int k1, int k2, int *ipv, int incX )
    {
        LAHPC_CHECK_POSITIVE( lda );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE_STRICT( incX );

        for ( int i = k1, xi = k1; i <= k2; ++i, xi += incX ) {
            int pivot = ipv[xi];
            if ( pivot != i ) {
                for ( int j = 0; j < N; ++j ) { std::swap( A[j * lda + i], A[j * lda + pivot] ); }
            }
        }
    }

} // namespace my_lapack
