#include "algonum.h"
#include "my_lapack.h"

#include <iostream>

using namespace my_lapack;

#define LAHPC_TESTALL( fct_t, testall_func, func ) \
    printf( " --- " #func ":\n" );                 \
    testall_func( static_cast<fct_t>( func ) );    \
    printf( "\n" );

void my_dgetrf_seq_test( CBLAS_ORDER order, int M, int N, double *A, int lda )
{
    if ( M == 0 || N == 0 ) { return; }

    const int nb    = 10;
    int       minMN = std::min( M, N );

    if ( nb <= 1 || nb >= minMN ) {
        my_dgetf2_seq( order, M, N, A, lda );
        return;
    }

    for ( int j = 0; j < minMN; j += nb ) {
        int jb = std::min( minMN - j, nb );
        my_dgetf2_seq( order, M - j, jb, A + j * lda + j, lda );
        if ( j + jb < N ) {
            my_dtrsm_seq( order,
                          CblasLeft,
                          CblasLower,
                          CblasNoTrans,
                          CblasUnit,
                          jb,
                          N - j - jb,
                          1.0,
                          A + j * lda + j,
                          lda,
                          A + ( j + jb ) * lda + j,
                          lda );

            if ( j + jb < M ) {
                my_dgemm_seq( CblasColMajor,
                              CblasNoTrans,
                              CblasNoTrans,
                              M - j - jb,
                              N - j - jb,
                              jb,
                              -1.0,
                              A + j * lda + j + jb,
                              lda,
                              A + ( j + jb ) * lda + j,
                              lda,
                              1.0,
                              A + ( j + jb ) * lda + j + jb,
                              lda );
            }
        }
    }
}

int main()
{
    printf( "----------- TEST ALGONUM -----------\n\n" );

    // M.Faverge's tests
    LAHPC_TESTALL( dgemm_fct_t, testall_dgemm, my_dgemm_scal_seq );

    LAHPC_TESTALL( dgemm_fct_t, testall_dgemm, my_dgemm_scal_openmp );

    LAHPC_TESTALL( dgemm_fct_t, testall_dgemm, my_dgemm_seq );

    LAHPC_TESTALL( dgemm_fct_t, testall_dgemm, my_dgemm_openmp );

    LAHPC_TESTALL( dgetrf_fct_t, testall_dgetrf, my_dgetf2_seq );

    LAHPC_TESTALL( dgetrf_fct_t, testall_dgetrf, my_dgetrf_seq_test );

    /*printf( "DGETRF OPENMP:\n" );
    testall_dgetrf( (dgetrf_fct_t) my_dgetrf_openmp );*/
    /* printf( "DGEMM TILED SEQUENTIAL:\n" );
    testall_dgetrf_tiled( (dgetrf_tiled_fct_t) tested_dgetrf );
    printf( "DGEMM TILED OPENMP:\n" );
    testall_dgemm_tiled( (dgemm_tiled_fct_t) tested_dgemm ); */

    return 0;
}
