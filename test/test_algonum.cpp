#include "algonum.h"
#include "my_lapack.h"

#include <iostream>

using namespace my_lapack;

#define LAHPC_TESTALL( fct_t, testall_func, func ) \
    printf( " --- " #func ":\n" );                 \
    testall_func( static_cast<fct_t>( func ) );    \
    printf( "\n" );

int main()
{
    printf( "----------- TEST ALGONUM -----------\n\n" );

    // M.Faverge's tests
    LAHPC_TESTALL( dgemm_fct_t, testall_dgemm, my_dgemm_scal_seq );

    LAHPC_TESTALL( dgemm_fct_t, testall_dgemm, my_dgemm_scal_openmp );

    LAHPC_TESTALL( dgemm_fct_t, testall_dgemm, my_dgemm_seq );

    LAHPC_TESTALL( dgemm_fct_t, testall_dgemm, my_dgemm_openmp );

    LAHPC_TESTALL( dgetrf_fct_t, testall_dgetrf, my_dgetf2_seq );

    LAHPC_TESTALL( dgetrf_fct_t, testall_dgetrf, my_dgetrf_seq );

    /*printf( "DGETRF OPENMP:\n" );
    testall_dgetrf( (dgetrf_fct_t) my_dgetrf_openmp );*/
    /* printf( "DGEMM TILED SEQUENTIAL:\n" );
    testall_dgetrf_tiled( (dgetrf_tiled_fct_t) tested_dgetrf );
    printf( "DGEMM TILED OPENMP:\n" );
    testall_dgemm_tiled( (dgemm_tiled_fct_t) tested_dgemm ); */

    return 0;
}
