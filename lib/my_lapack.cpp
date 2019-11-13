#include "my_lapack.h"

#include "err.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#define AT_RM(i,j,width)  ((i)*(width) + (j))
#define AT(i,j,heigth) ((j)*(heigth) + (i))

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
    
    /// M N and K aren't changed even if transposed.
    void my_dgemm_scalaire( const enum LAHPC_ORDER     Order,
                            const enum LAHPC_TRANSPOSE TransA,
                            const enum LAHPC_TRANSPOSE TransB,
                            const int                  M,
                            const int                  N,
                            const int                  K,
                            const double               alpha,
                            const double *             A,
                            const int                  lda,
                            const double *             B,
                            const int                  ldb,
                            const double               beta,
                            double *                   C,
                            const int                  ldc )
    {
        (void) Order;
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( K );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( ldb );
        LAHPC_CHECK_POSITIVE_STRICT( ldc );

        // Initializing C matrix with 0 (could be removed if yoou want to keep what is already inside C)
        memset(C, 0, sizeof(double)*ldc);

        // Calculating dgemm
        bool bTransA = (TransA == LAHPC_TRANSPOSE::Trans);
        bool bTransB = (TransB == LAHPC_TRANSPOSE::Trans);
        size_t i,j,k;
        if (bTransA && bTransB){
            for(i = 0; i < M; i++){
                for(j = 0; j < N; j++){
                    for(k = 0; k < K; k++){
                        C[AT(i, j, ldc)] += alpha*A[AT(j, i, lda)] + beta*B[AT(j, i, ldb)];
                    }
                }
            }
        } else if (!bTransA && bTransB){
            for(i = 0; i < M; i++){
                for(j = 0; j < N; j++){
                    for(k = 0; k < K; k++){
                        C[AT(i, j, ldc)] += alpha*A[AT(i, j, lda)] + beta*B[AT(j, i, ldb)];
                    }
                }
            }
        } else if (bTransA && !bTransB){
            for(i = 0; i < M; i++){
                for(j = 0; j < N; j++){
                    for(k = 0; k < K; k++){
                        C[AT(i, j, ldc)] += alpha*A[AT(j, i, lda)] + beta*B[AT(i, j, ldb)];
                    }
                }
            }
        } else {
            for(i = 0; i < M; i++){
                for(j = 0; j < N; j++){
                    for(k = 0; k < K; k++){
                        C[AT(i, j, ldc)] += alpha*A[AT(i, j, lda)] + beta*B[AT(i, j, ldb)];
                    }
                }
            }
        }
    }

    void my_dgemm( const enum LAHPC_ORDER     Order,
                   const enum LAHPC_TRANSPOSE TransA,
                   const enum LAHPC_TRANSPOSE TransB,
                   const int                  M,
                   const int                  N,
                   const int                  K,
                   const double               alpha,
                   const double *             A,
                   const int                  lda,
                   const double *             B,
                   const int                  ldb,
                   const double               beta,
                   double *                   C,
                   const int                  ldc )
    {
        
    }

    void my_dger(const enum LAHPC_ORDER order,
                    const int M,
                    const int N,
                    const double alpha,
                    const double *X,
                    const int incX,
                    const double *Y,
                    const int incY,
                    double *A,
                    const int lda)
    {
        (void) order;
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE_STRICT( lda );
        LAHPC_CHECK_POSITIVE_STRICT( ldb );
        LAHPC_CHECK_POSITIVE_STRICT( incX );
        LAHPC_CHECK_POSITIVE_STRICT( incY );

        size_t i,j;
        for(i = 0; i < M; i += incX){
            for(j = 0; j < N; j += incY){
                A[AT(i, j, lda)] += alpha*X[i]*Y[j];
            }   
        }
    }



} // namespace my_lapack