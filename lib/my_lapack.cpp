#include "my_lapack.h"

#include "err.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

static const int BLOCK_SIZE = 128;

#define AT_RM( i, j, width ) (( i ) * ( width ) + ( j ))
#define AT( i, j, heigth ) (( j ) * ( heigth ) + ( i ))
#define min_macro( a, b ) (( a ) < ( b ) ? ( a ) : ( b ))

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

        if ( alpha == 0.0 ) { return; }

        for ( int i = 0, xi = 0, yi = 0; i < N; ++i, xi += incX, yi += incY ) { Y[yi] += alpha * X[xi]; }
    }

    void my_dgemv( enum CBLAS_ORDER layout,
                   enum CBLAS_TRANSPOSE TransA,
                   int M,
                   int N,
                   double alpha,
                   const double * A,
                   int lda,
                   const double * X,
                   int incX,
                   double beta,
                   double * Y,
                   const int incY )
    {
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE_STRICT( incX );
        LAHPC_CHECK_POSITIVE_STRICT( incY );
        LAHPC_CHECK_PREDICATE( layout == CblasColMajor );

        if ( M == 0 || N == 0 || ( alpha == 0.0 && beta == 1.0 )) return;

        if ( beta != 1.0 ) {
            int lenY = ( TransA == CblasNoTrans ) ? M : N;

            if ( beta == 0 && incY == 1 ) { memset( Y, 0, lenY * sizeof( double )); }
            else if ( beta == 0 ) {
                for ( int i = 0, yi = 0; i < lenY; ++i, yi += incY ) { Y[yi] = 0; }
            }
            else {
                for ( int i = 0, yi = 0; i < lenY; ++i, yi += incY ) { Y[yi] *= beta; }
            }
        }

        if ( TransA == CblasNoTrans ) {
            for ( int j = 0, xi = 0; j < N; ++j, xi += incX ) {
                double tmp = alpha * X[xi];
                for ( int i = 0, yi = 0; i < M; ++i, yi += incY ) { Y[yi] += tmp * A[j * M + i]; }
            }
        }

        else if ( TransA == CblasTrans ) {
            for ( int j = 0, yi = 0; j < N; ++j, yi += incY ) {
                double tmp = 0;
                for ( int i = 0, xi = 0; i < N; ++i, xi += incX ) { tmp += A[j * M + i] * X[xi]; }
                Y[yi] += alpha * tmp;
            }
        }
    }

    /// M N and K aren't changed even if transposed.
    void my_dgemm_scalaire( enum CBLAS_ORDER layout,
                            enum CBLAS_TRANSPOSE TransA,
                            enum CBLAS_TRANSPOSE TransB,
                            int M,
                            int N,
                            int K,
                            double alpha,
                            const double * A,
                            int lda,
                            const double * B,
                            int ldb,
                            double beta,
                            double * C,
                            int ldc )
    {
        LAHPC_CHECK_PREDICATE( layout == CblasColMajor );
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( K );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( ldb );
        LAHPC_CHECK_POSITIVE_STRICT( ldc );

        // Early return
        if ( alpha == 0. ) {
            if ( beta != 1. ) {
                size_t m, n;
                for (m = 0; m < M; m++){
                    for (n = 0; n < N; n++){
                        C[AT(m,n,ldc)] += beta*C[AT(m,n,ldc)];
                    }
                }
            }
            return;
        }

        //
        bool bTransA = ( TransA == CBLAS_TRANSPOSE::CblasTrans );
        bool bTransB = ( TransB == CBLAS_TRANSPOSE::CblasTrans );

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
                    C[AT( i, j, ldc )] *= beta;
                    for ( k = 0; k < M; k++ ) {
                        C[AT( i, j, ldc )] +=
                            alpha * A[AT( j, k, lda )] * B[AT( k, i, ldb )] + C[AT( i, j, ldc )];
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
                    C[AT( i, j, ldc )] *= beta;
                    for ( k = 0; k < K; k++ ) {
                        C[AT( i, j, ldc )] +=
                            alpha * A[AT( i, k, lda )] * B[AT( k, i, ldb )] + C[AT( i, j, ldc )];
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
                    C[AT( i, j, ldc )] *= beta;
                    for ( k = 0; k < K; k++ ) {
                        C[AT( i, j, ldc )] +=
                            alpha * A[AT( j, k, lda )] * B[AT( k, j, ldb )] + C[AT( i, j, ldc )];
                    }
                }
            }
        }
        else {
            for ( j = 0; j < N; j++ ) {
                for ( i = 0; i < M; i++ ) {
                    C[AT( i, j, ldc )] *= beta;
                    for ( k = 0; k < K; k++ ) {
                        C[AT( i, j, ldc )] +=
                            alpha * A[AT( i, k, lda )] * B[AT( k, j, ldb )] + C[AT( i, j, ldc )];
                    }
                }
            }
        }
    }

    void my_dgemm( enum CBLAS_ORDER    Order,
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
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( K );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( ldb );
        LAHPC_CHECK_POSITIVE_STRICT( ldc );

        int blocksize = std::min( std::min( M, N ), BLOCK_SIZE );

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

    void my_dger( enum CBLAS_ORDER layout,
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
        LAHPC_CHECK_PREDICATE( layout == CblasColMajor );
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

    void my_dgetrf2( int M, int N, double *A, int lda, int *ipiv, int *info )
    {
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_PREDICATE( lda >= std::max( 1, M ) );

        *info = 0;

        if ( M == 0 || N == 0 ) { return; }

        if ( M == 1 ) {
            ipiv[0] = 1;
            if ( A[0] == 0.0 ) { *info = 1; }
        }
        else if ( N == 1 ) {
            int pivot = my_idamax( M, A, 1 );
            ipiv[0]   = pivot;
            if ( A[pivot] != 0.0 ) {
                if ( pivot != 0 ) { std::swap( A[0], A[pivot] ); }

                if ( std::abs( A[0] ) > std::numeric_limits<double>::epsilon() ) {
                    my_dscal( M - 1, 1.0 / A[0], A + 1, 1 );
                }
                else {
                    for ( int i = 1; i < M; ++i ) { A[i] /= A[0]; } // Can't see why it would not explode...
                }
            }
            else {
                *info = 1;
            }
        }
        else {
            int n1 = std::min( M, N ) / 2;
            int n2 = N - n1;

            int infoRec;
            my_dgetrf2( M, n1, A, lda, ipiv, &infoRec );

            if ( *info == 0 && infoRec > 0 ) { *info = infoRec; }

            my_dlaswp( n2,
                       A + static_cast<std::size_t>( n1 ) * static_cast<std::size_t>( lda ),
                       lda,
                       0,
                       n1 - 1,
                       ipiv,
                       1 );


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
            float tmp = std::abs( dx[i] );
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
