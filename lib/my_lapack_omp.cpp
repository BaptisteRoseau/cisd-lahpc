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

#define _LAHPC_BLOCK_SIZE 34
static const int BLOCK_SIZE = _LAHPC_BLOCK_SIZE;

#define AT_RM( i, j, width ) ( ( i ) * ( width ) + ( j ) )
#define AT( i, j, heigth ) ( ( j ) * ( heigth ) + ( i ) )
#define min_macro( a, b ) ( ( a ) < ( b ) ? ( a ) : ( b ) )

// TODO: simd if incX == 1

namespace my_lapack {

    double my_ddot_openmp( const int N, const double *X, const int incX, const double *Y, const int incY )
    {
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( incX );
        LAHPC_CHECK_POSITIVE( incY );

        double ret = 0;
        int xi, yi;
        #pragma omp parallel for reduction(+:ret) default(shared) private(xi, yi)
        for ( int i = 0; i < N; ++i) {
            xi = i*incX;
            yi = i*incY;
            ret += X[xi] * Y[yi];
        }
        return ret;
    }

    void my_daxpy_openmp( const int N, const double alpha, const double *X, const int incX, double *Y, const int incY )
    {
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( incX );
        LAHPC_CHECK_POSITIVE( incY );

        if ( alpha == 0.0 ) { return; }

            int xi, yi;
#pragma omp parallel for private(xi, yi) default( shared )
            for ( int i = 0; i < N; i++ ) {
                yi = i * incY;
                xi = i * incX;
                Y[yi] += alpha * X[xi];
            }
    }

    void my_dgemv_openmp( CBLAS_ORDER     layout,
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
        LAHPC_CHECK_PREDICATE( ( TransA == CblasTrans ) || ( TransA == CblasNoTrans ) );

        if ( M == 0 || N == 0 || ( alpha == 0.0 && beta == 1.0 ) ) return;

        if ( beta != 1.0 ) {
            int lenY = ( TransA == CblasNoTrans ) ? M : N;

            if ( beta == 0 && incY == 1 ) { memset( Y, 0, lenY * sizeof( double ) ); }
            else if ( beta == 0 ) {
                int len = lenY * incY;
#pragma omp parallel for default( shared )
                for ( int yi = 0; yi < len; yi += incY ) {
                    Y[yi] = 0;
                }
            }
            else {
                int len = lenY * incY;
#pragma omp parallel for default( shared )
                for ( int yi = 0; yi < len; yi += incY ) {
                    Y[yi] *= beta;
                }
            }
        }

        if ( TransA == CblasNoTrans ) {
#pragma omp parallel for default( shared )
            for ( int j = 0; j < N; ++j ) {
                double tmp = alpha * X[incX * j];
                for ( int i = 0; i < M; i++ ) {
                    Y[i * incY] += tmp * A[j * lda + i];
                }
            }
        }

        else if ( TransA == CblasTrans ) {
#pragma omp parallel for default( shared )
            for ( int j = 0; j < N; ++j ) {
                int    yi  = j * incY;
                double tmp = 0;
                for ( int i = 0, xi = 0; i < M; ++i, xi += incX ) {
                    tmp += A[j * lda + i] * X[xi];
                }
                Y[yi] += alpha * tmp;
            }
        }
    }

    // TODO: reduce on linear add
    void my_dgemm_scal_openmp( CBLAS_ORDER     Order,
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
        LAHPC_CHECK_PREDICATE( ( TransA == CblasTrans ) || ( TransA == CblasNoTrans ) );
        LAHPC_CHECK_PREDICATE( ( TransB == CblasTrans ) || ( TransB == CblasNoTrans ) );
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( K );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( ldb );
        LAHPC_CHECK_POSITIVE_STRICT( ldc );

        // Early return
        if ( alpha == 0. ) {
            if ( beta != 1. ) {
                int i, len = M * N;
#pragma omp parallel for simd
                for ( i = 0; i < len; i++ ) {
                    C[i] *= beta;
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
                    for ( k = 0; k < K; k++ ) {
                        C[AT( m, n, ldc )] += alpha * A[AT( k, m, lda )] * B[AT( n, k, ldb )];
                    }
                }
            }
        }
        else if ( !bTransA && bTransB ) {
#pragma omp parallel for default( shared ) collapse( 2 ) private( k )
            for ( n = 0; n < N; n++ ) {
                for ( m = 0; m < M; m++ ) {
                    C[AT( m, n, ldc )] *= beta;
                    for ( k = 0; k < K; k++ ) {
                        C[AT( m, n, ldc )] += alpha * A[AT( m, k, lda )] * B[AT( n, k, ldb )];
                    }
                }
            }
        }
        else if ( bTransA && !bTransB ) {
#pragma omp parallel for default( shared ) collapse( 2 ) private( k )
            for ( n = 0; n < N; n++ ) {
                for ( m = 0; m < M; m++ ) {
                    C[AT( m, n, ldc )] *= beta;
                    for ( k = 0; k < K; k++ ) {
                        C[AT( m, n, ldc )] += alpha * A[AT( k, m, lda )] * B[AT( k, n, ldb )];
                    }
                }
            }
        }
        else {
#pragma omp parallel for default( shared ) collapse( 2 ) private( k )
            for ( n = 0; n < N; n++ ) {
                for ( m = 0; m < M; m++ ) {
                    C[AT( m, n, ldc )] *= beta;
                    for ( k = 0; k < K; k++ ) {
                        C[AT( m, n, ldc )] += alpha * A[AT( m, k, lda )] * B[AT( k, n, ldb )];
                    }
                }
            }
        }
    }

    void my_dgemm_openmp( CBLAS_ORDER     Order,
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
        LAHPC_CHECK_PREDICATE( ( TransA == CblasTrans ) || ( TransA == CblasNoTrans ) );
        LAHPC_CHECK_PREDICATE( ( TransB == CblasTrans ) || ( TransB == CblasNoTrans ) );
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( K );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( ldb );
        LAHPC_CHECK_POSITIVE_STRICT( ldc );

        // Early return
        if ( alpha == 0. && beta == 1. ) { return; }
        if ( alpha == 0 ) {
            my_dgemm_scal_openmp( Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc );
            return;
        }

        // Computing booleans in advance
        bool bTransA = ( TransA == CblasTrans );
        bool bTransB = ( TransB == CblasTrans );

        int lastMB = M % BLOCK_SIZE;
        int lastNB = N % BLOCK_SIZE;
        int lastKB = K % BLOCK_SIZE;
        int MB     = lastMB ? ( M / BLOCK_SIZE ) + 1 : ( M / BLOCK_SIZE );
        int NB     = lastNB ? ( N / BLOCK_SIZE ) + 1 : ( N / BLOCK_SIZE );
        int KB     = lastKB ? ( K / BLOCK_SIZE ) + 1 : ( K / BLOCK_SIZE );

        double *C_padding;
        int     m, n, k, m_blk, n_blk;
        if ( bTransA && bTransB ) {
#pragma omp parallel for default( shared ) private( m, n, k, m_blk, n_blk, C_padding )
            for ( m = 0; m < MB; m++ ) {
                m_blk = m < MB - 1 ? BLOCK_SIZE : lastMB;
#pragma omp parallel for default( shared ) private( n, k, n_blk, C_padding )
                for ( n = 0; n < NB; n++ ) {
                    n_blk     = n < NB - 1 ? BLOCK_SIZE : lastNB;
                    C_padding = C + BLOCK_SIZE * AT( m, n, ldc );
#pragma omp parallel for default( shared )
                    for ( int l = 0; l < m_blk; ++l ) {
                        for ( int c = 0; c < n_blk; ++c ) {
                            C_padding[AT( l, c, ldc )] *= beta;
                        }
                    }
#pragma omp parallel for default( shared ) schedule(dynamic, 4)
                    for ( k = 0; k < KB; k++ ) {
                        my_dgemm_scal_seq( Order,
                                              TransA,
                                              TransB,
                                              m_blk,
                                              n_blk,
                                              k < KB - 1 ? BLOCK_SIZE : lastKB,
                                              alpha,
                                              A + BLOCK_SIZE * AT( k, m, lda ),
                                              lda,
                                              B + BLOCK_SIZE * AT( n, k, ldb ),
                                              ldb,
                                              1.,
                                              C_padding,
                                              ldc );
                    }
                }
            }
        }
        else if ( !bTransA && bTransB ) {
#pragma omp parallel for default( shared ) private( m, n, k, m_blk, n_blk, C_padding )
            for ( m = 0; m < MB; m++ ) {
                m_blk = m < MB - 1 ? BLOCK_SIZE : lastMB;
#pragma omp parallel for default( shared ) private( n, k, n_blk, C_padding )
                for ( n = 0; n < NB; n++ ) {
                    n_blk     = n < NB - 1 ? BLOCK_SIZE : lastNB;
                    C_padding = C + BLOCK_SIZE * AT( m, n, ldc );
#pragma omp parallel for default( shared )
                    for ( int l = 0; l < m_blk; ++l ) {
                        for ( int c = 0; c < n_blk; ++c ) {
                            C_padding[AT( l, c, ldc )] *= beta;
                        }
                    }
#pragma omp parallel for default( shared ) schedule(dynamic, 4)
                    for ( k = 0; k < KB; k++ ) {
                        my_dgemm_scal_seq( Order,
                                              TransA,
                                              TransB,
                                              m_blk,
                                              n_blk,
                                              k < KB - 1 ? BLOCK_SIZE : lastKB,
                                              alpha,
                                              A + BLOCK_SIZE * AT( m, k, lda ),
                                              lda,
                                              B + BLOCK_SIZE * AT( n, k, ldb ),
                                              ldb,
                                              1.,
                                              C_padding,
                                              ldc );
                    }
                }
            }
        }
        else if ( bTransA && !bTransB ) {
#pragma omp parallel for default( shared ) private( m, n, k, m_blk, n_blk, C_padding )
            for ( m = 0; m < MB; m++ ) {
                m_blk = m < MB - 1 ? BLOCK_SIZE : lastMB;
#pragma omp parallel for default( shared ) private( n, k, n_blk, C_padding )
                for ( n = 0; n < NB; n++ ) {
                    n_blk     = n < NB - 1 ? BLOCK_SIZE : lastNB;
                    C_padding = C + BLOCK_SIZE * AT( m, n, ldc );
#pragma omp parallel for default( shared )
                    for ( int l = 0; l < m_blk; ++l ) {
                        for ( int c = 0; c < n_blk; ++c ) {
                            C_padding[AT( l, c, ldc )] *= beta;
                        }
                    }
#pragma omp parallel for default( shared ) schedule(dynamic, 4)
                    for ( k = 0; k < KB; k++ ) {
                        my_dgemm_scal_seq( Order,
                                              TransA,
                                              TransB,
                                              m_blk,
                                              n_blk,
                                              k < KB - 1 ? BLOCK_SIZE : lastKB,
                                              alpha,
                                              A + BLOCK_SIZE * AT( k, m, lda ),
                                              lda,
                                              B + BLOCK_SIZE * AT( k, n, ldb ),
                                              ldb,
                                              1.,
                                              C_padding,
                                              ldc );
                    }
                }
            }
        }
        else {
#pragma omp parallel for default( shared ) private( m, n, k, m_blk, n_blk, C_padding )
            for ( m = 0; m < MB; m++ ) {
                m_blk = m < MB - 1 ? BLOCK_SIZE : lastMB;
#pragma omp parallel for default( shared ) private( n, k, n_blk, C_padding )
                for ( n = 0; n < NB; n++ ) {
                    n_blk     = n < NB - 1 ? BLOCK_SIZE : lastNB;
                    C_padding = C + BLOCK_SIZE * AT( m, n, ldc );
#pragma omp parallel for default( shared )
                    for ( int l = 0; l < m_blk; ++l ) {
                        for ( int c = 0; c < n_blk; ++c ) {
                            C_padding[AT( l, c, ldc )] *= beta;
                        }
                    }
#pragma omp parallel for default( shared ) schedule(dynamic, 4)
                    for ( k = 0; k < KB; k++ ) {
                        my_dgemm_scal_seq( Order,
                                              TransA,
                                              TransB,
                                              m_blk,
                                              n_blk,
                                              k < KB - 1 ? BLOCK_SIZE : lastKB,
                                              alpha,
                                              A + BLOCK_SIZE * AT( m, k, lda ),
                                              lda,
                                              B + BLOCK_SIZE * AT( k, n, ldb ),
                                              ldb,
                                              1.,
                                              C_padding,
                                              ldc );
                    }
                }
            }
        }
    }

    void my_dger_openmp( CBLAS_ORDER   layout,
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

        for ( int i = 0; i < M; ++i ) {
            double tmp = alpha * X[i * incX];
            for ( int j = 0; j < N; ++j ) {
                A[i + j * lda] += tmp * Y[j * incY];
            }
        }
    }

    void my_dgetf2_openmp( CBLAS_ORDER order, int M, int N, double *A, int lda )
    {
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_PREDICATE( lda >= std::max( 1, M ) );

        if ( M == 0 || N == 0 ) { return; }

        int minMN = std::min( M, N );
        for ( int j = 0; j < minMN; ++j ) {
            if ( j < M - 1 ) { my_dscal( M - j - 1, 1.0 / A[j * lda + j], A + j * lda + j + 1, 1 ); }
            if ( j < minMN - 1 ) {
                my_dger_openmp( CblasColMajor,
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

    void my_dtrsm_openmp( CBLAS_ORDER     layout,
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
        LAHPC_CHECK_PREDICATE( layout == CblasColMajor );
        LAHPC_CHECK_PREDICATE( ( transA == CblasTrans ) || ( transA == CblasNoTrans ) );
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( ldb );

        double lambda;

        if ( M == 0 || N == 0 ) return;

        /* scale 0. */
        if ( alpha == 0. ) {
            for ( int j = 0; j < N; ++j ) {
                memset( B + j * ldb, 0, M * sizeof( double ) );
            }
            return;
        }

        /* Left side : op( A ) * X = alpha * B */
        if ( side == CblasLeft ) {
            /* B = alpha * inv(A ** t) * B */
            if ( transA == CblasTrans ) {
                /* A is a lower triangular */
                if ( uplo == CblasLower ) {
#pragma omp parallel for default( none ) private( lambda ) \
    shared( layout, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb )
                    for ( int j = 0; j < N; ++j ) {
                        for ( int i = M - 1; i >= 0; --i ) {
                            lambda = alpha * B[i + j * ldb];
                            for ( int k = i + 1; k < M; ++k ) {
                                lambda -= B[k + j * ldb] * A[k + i * lda];
                            }
                            /* The diagonal is A[i + i*lda] (Otherwise : 1.) */
                            /* Relevent when solving A = L*U as we use A to store
                               both L and U, so diag(L) is full of 1. . */
                            if ( diag == CblasNonUnit ) lambda /= A[i * ( 1 + lda )];
                            B[i + j * ldb] = lambda;
                        }
                    }
                }
                /* A is triangular upper */
                else if ( uplo == CblasUpper ) {
#pragma omp parallel for default( none ) private( lambda ) \
    shared( layout, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb )
                    for ( int j = 0; j < N; ++j ) {
                        for ( int i = 0; i < M; ++i ) {
                            lambda = alpha * B[i + j * ldb];
                            for ( int k = 0; k < i; ++k ) {
                                lambda -= A[k + i * lda] * B[k + j * ldb];
                            }
                            /* The diagonal is A[i + i*lda] (Otherwise : 1.) */
                            if ( diag == CblasNonUnit ) lambda /= A[i * ( 1 + lda )];
                            B[i + j * ldb] = lambda;
                        }
                    }
                }
            }
            /* B = alpha * inv(A) * B */
            else {
                /* A is triangular Upper */
                if ( uplo == CblasUpper ) {
#pragma omp parallel for default( none ) private( lambda ) \
    shared( layout, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb )
                    for ( int j = 0; j < N; ++j ) {
                        if ( alpha != 1. ) {
                            for ( int i = 0; i < M; i++ ) {
                                B[i + j * ldb] *= alpha;
                            }
                        }
                        for ( int k = M - 1; k >= 0; --k ) {
                            if ( B[k + j * ldb] ) {
                                if ( diag == CblasNonUnit ) B[k + j * ldb] /= A[k * ( 1 + lda )];
                                lambda = B[k + j * ldb];
                                for ( int i = 0; i < k; ++i ) {
                                    B[i + j * ldb] -= lambda * A[i + k * lda];
                                }
                            }
                        }
                    }
                }
                /* A is lower triangular */
                else {
#pragma omp parallel for default( none ) private( lambda ) \
    shared( layout, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb )
                    for ( int j = 0; j < N; ++j ) {
                        for ( int i = 0; i < M; i++ ) {
                            B[i + j * ldb] *= alpha;
                        }
                        for ( int k = 0; k < M; ++k ) {
                            if ( B[k + j * ldb] != 0. ) {
                                if ( diag == CblasNonUnit ) B[k + j * ldb] /= A[k * ( 1 + lda )];
                                lambda = B[k + j * ldb];
                                for ( int i = k + 1; i < M; ++i ) {
                                    B[i + j * ldb] -= lambda * A[i + k * lda];
                                }
                            }
                        }
                    }
                }
            }
        }
        /* Right side : X * op( A ) = alpha*B */
        else {
            /* X = alpha * B * inv(A) */
            if ( transA == CblasNoTrans ) {
                /* A is upper triangular */
                if ( uplo == CblasUpper ) {
#pragma omp parallel for default( none ) private( lambda ) \
    shared( layout, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb )
                    for ( int j = 0; j < N; j++ ) {
                        if ( alpha != 1.0 ) {
                            for ( int i = 0; i < M; ++i ) {
                                B[i + j * ldb] *= alpha;
                            }
                        }
                        for ( int k = 0; k < j - 1; k++ ) {
                            if ( A[k + j * lda] != 0.0 ) {
                                for ( int i = 0; i < M; i++ ) {
                                    B[i + j * ldb] -= A[k + j * lda] * B[i + k * ldb];
                                }
                            }
                        }
                        if ( diag == CblasNonUnit ) {
                            lambda = 1.0 / A[j * ( 1 + lda )];
                            for ( int i = 0; i < M; i++ ) {
                                B[i + j * ldb] = lambda * B[i + j * ldb];
                            }
                        }
                    }
                }
                /* A is lower triangular */
                else {
#pragma omp parallel for default( none ) private( lambda ) \
    shared( layout, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb )
                    for ( int j = N - 1; j >= 0; --j ) {
                        if ( alpha != 1.0 ) {
                            for ( int i = 0; i < M; ++i ) {
                                B[i + j * ldb] *= alpha;
                            }
                        }
                        for ( int k = j + 1; k < N; ++k ) {
                            if ( A[k + j * lda] != 0.0 ) {
                                for ( int i = 0; i < M; ++i ) {
                                    B[i + j * ldb] -= A[k + j * lda] * B[i + k * ldb];
                                }
                            }
                        }
                        if ( diag == CblasNonUnit ) {
                            lambda = 1.0 / A[j * ( 1 + lda )];
                            for ( int i = 0; i < M; i++ ) {
                                B[i + j * ldb] = lambda * B[i + j * ldb];
                            }
                        }
                    }
                }
            }
            /* X = alpha * B * inv(A ** t) */
            else {
                /* A is upper triangular */
                if ( uplo == CblasUpper ) {
#pragma omp parallel for default( none ) private( lambda ) \
    shared( layout, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb )
                    for ( int k = N - 1; k >= 0; --k ) {
                        if ( diag == CblasNonUnit ) {
                            lambda = 1.0 / A[k + k * lda];
                            for ( int i = 0; i < M; i++ ) {
                                B[i + k * ldb] = lambda * B[i + k * ldb];
                            }
                        }
                        for ( int j = 0; j < k; ++j ) {
                            if ( A[j + k * lda] != 0.0 ) {
                                lambda = A[j + k * lda];
                                for ( int i = 0; i < M; ++i ) {
                                    B[i + j * ldb] -= lambda * B[i + k * ldb];
                                }
                            }
                        }
                        if ( alpha != 1.0 ) {
                            for ( int i = 0; i < M; i++ ) {
                                B[i + k * ldb] = alpha * B[i + k * ldb];
                            }
                        }
                    }
                }
                /* A is lower triangular */
                else {
#pragma omp parallel for default( none ) private( lambda ) \
    shared( layout, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb )
                    for ( int k = 0; k < N; ++k ) {
                        if ( diag == CblasNonUnit ) {
                            lambda = 1.0 / A[k + k * lda];
                            for ( int i = 0; i < M; ++i ) {
                                B[i + k * ldb] = lambda * B[i + k * ldb];
                            }
                        }
                        for ( int j = k + 1; j < N; j++ ) {
                            if ( A[j + k * lda] != 0.0 ) {
                                lambda = A[j + k * lda];
                                for ( int i = 0; i < M; i++ ) {
                                    B[i + j * ldb] -= lambda * B[i + k * ldb];
                                }
                            }
                        }
                        if ( alpha != 1.0 ) {
                            for ( int i = 0; i < M; ++i ) {
                                B[i + k * lda] = alpha * B[i + k * ldb];
                            }
                        }
                    }
                }
            }
        }
    }

    void my_dgetrf_openmp( CBLAS_ORDER order, int M, int N, double *A, int lda )
    {
        LAHPC_CHECK_PREDICATE( order == CblasColMajor );
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE_STRICT( lda );

        //if ( M == 0 || N == 0 ) { return; }

        const int maxBlockSize = 10;
        int       minMN        = std::min( M, N );

        if ( maxBlockSize <= 1 || maxBlockSize >= minMN ) {
            my_dgetf2_seq( order, M, N, A, lda );
            return;
        }
//#pragma omp parallel for schedule( guided ) default( shared )
        for ( int j = 0; j < minMN; j += maxBlockSize ) {
            int blockSize = std::min( minMN - j, maxBlockSize );
            my_dgetrf_openmp( order, M - j, blockSize, A + j * lda + j, lda );
            if ( j + blockSize < N ) {
                my_dtrsm_openmp( order,
                                 CblasLeft,
                                 CblasLower,
                                 CblasNoTrans,
                                 CblasUnit,
                                 blockSize,
                                 N - j - blockSize,
                                 1.0,
                                 A + j * lda + j,
                                 lda,
                                 A + ( j + blockSize ) * lda + j,
                                 lda );

                if ( j + blockSize < M ) {
                    my_dgemm_openmp( CblasColMajor,
                                     CblasNoTrans,
                                     CblasNoTrans,
                                     M - j - blockSize,
                                     N - j - blockSize,
                                     blockSize,
                                     -1.0,
                                     A + j * lda + j + blockSize,
                                     lda,
                                     A + ( j + blockSize ) * lda + j,
                                     lda,
                                     1.0,
                                     A + ( j + blockSize ) * lda + j + blockSize,
                                     lda );
                }
            }
        }
    }

    int my_idamax_openmp( int N, double *dx, int incX )
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

    void my_dscal_openmp( int N, double da, double *dx, int incX )
    {
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE_STRICT( incX );

        if ( N == 0 ) { return; }
        if ( da == 0.0 && incX == 1 ) {
            std::memset( dx, 0, N * sizeof( double ) );
            return;
        }
        if ( da == 0.0 ) {
            for ( int i = 0, xi = 0; i < N; ++i, xi += incX ) {
                dx[xi] = 0.0;
            }
        }
        else {
            for ( int i = 0, xi = 0; i < N; ++i, xi += incX ) {
                dx[xi] *= da;
            }
        }
    }

    void my_dlaswp_openmp( int N, double *A, int lda, int k1, int k2, int *ipv, int incX )
    {
        LAHPC_CHECK_POSITIVE( lda );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE_STRICT( incX );

        for ( int i = k1, xi = k1; i <= k2; ++i, xi += incX ) {
            int pivot = ipv[xi];
            if ( pivot != i ) {
                for ( int j = 0; j < N; ++j ) {
                    std::swap( A[j * lda + i], A[j * lda + pivot] );
                }
            }
        }
    }

    // TODO: Implement these ones
    /* void my_dgemm_tiled_openmp( CBLAS_LAYOUT layout,
                                CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                                int M, int N, int K, int b,
                                double alpha, const double **A,
                                              const double **B,
                                double beta,        double **C );

    void my_dgetrf_tiled_openmp( CBLAS_LAYOUT layout,
                                 int m, int n, int b, double **a );
     */

} // namespace my_lapack
