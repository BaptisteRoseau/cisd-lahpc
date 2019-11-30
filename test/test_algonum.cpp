#include "algonum.h"
#include "my_lapack.h"

#include <iostream>

using namespace my_lapack;

int main()
{
    printf( "----------- TEST ALGONUM -----------\n" );

    // M.Faverge's tests
    printf( "DGEMM SCAL SEQUENTIAL:\n" );
    testall_dgemm( static_cast<dgemm_fct_t>( my_dgemm_scal_seq ) );

    printf( "DGEMM SCAL OPENMP:\n" );
    testall_dgemm( static_cast<dgemm_fct_t>( my_dgemm_scal_openmp ) );

    printf( "DGEMM SEQUENTIAL:\n" );
    testall_dgemm( static_cast<dgemm_fct_t>( my_dgemm_seq ) );

    printf( "DGEMM OPENMP:\n" );
    testall_dgemm( static_cast<dgemm_fct_t>( my_dgemm_openmp ) );

    printf( "DGETRF SEQUENTIAL:\n" );
    testone_dgetrf( static_cast<dgetrf_fct_t>( my_dgetrf_seq ), 100, 100, 0 );
    testall_dgetrf( static_cast<dgetrf_fct_t>( my_dgetrf_seq ) );

    /*printf( "DGETRF OPENMP:\n" );
    testall_dgetrf( (dgetrf_fct_t) my_dgetrf_openmp );*/
    /* printf( "DGEMM TILED SEQUENTIAL:\n" );
    testall_dgetrf_tiled( (dgetrf_tiled_fct_t) tested_dgetrf );
    printf( "DGEMM TILED OPENMP:\n" );
    testall_dgemm_tiled( (dgemm_tiled_fct_t) tested_dgemm ); */

    return 1;
}
