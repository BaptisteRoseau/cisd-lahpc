#include "Summa.hpp"
#include "err.h"
#include "my_lapack.h"

#include <algorithm>
#include <mpi.h>

/* Access matrix elements provided storage is COLUMN MAJOR */
#define A( i, j ) ( a[j * lda + i] )
#define B( i, j ) ( b[j * ldb + i] )
#define C( i, j ) ( c[j * ldc + i] )

static Summa SUMMA;

void my_dlacpy( int M, int N, const double *a, int lda, double *b, int ldb )
{
    LAHPC_CHECK_POSITIVE( M );
    LAHPC_CHECK_POSITIVE( N );
    LAHPC_CHECK_PREDICATE( lda >= std::max( 1, M ) );
    LAHPC_CHECK_PREDICATE( ldb >= std::max( 1, M ) );

    for ( int j = 0; j < N; ++j ) {
        for ( int i = 0; i < M; ++i ) {
            B( i, j ) = A( i, j );
        }
    }
}

void init_lapack_mpi(int* argc, char*** argv)
{
    SUMMA.init( argc, argv );
}

void my_dgemm_mpi( CBLAS_ORDER     Order,
                   CBLAS_TRANSPOSE TransA,
                   CBLAS_TRANSPOSE TransB,
                   int             M,
                   int             N,
                   int             K,
                   double          alpha,
                   const double *  a,
                   int             lda,
                   const double *  b,
                   int             ldb,
                   double          beta,
                   double *        c,
                   int             ldc )
{
    SUMMA.reset(M, N, K);



    SUMMA.finalize();
}
