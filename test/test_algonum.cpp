#include "algonum.h"
#include "my_lapack.h"

using namespace my_lapack;

#ifdef __cplusplus
extern "C"{
#endif

int main( int argc, char **argv )
{
    printf( "----------- TEST ALGONUM -----------\n" );

    // M.Faverge's tests
    printf( "DGEMM OPENMP:\n" );
    testall_dgemm( (dgemm_fct_t) my_dgemm_openmp );
    printf( "DGEMM SEQUENTIAL:\n" );
    testall_dgemm( (dgemm_fct_t) my_dgemm_seq );
    printf( "DGETRF OPENMP:\n" );
    testall_dgetrf( (dgetrf_fct_t) my_dgetrf_openmp );
    printf( "DGETRF SEQUENTIAL:\n" );
    testall_dgetrf( (dgetrf_fct_t) my_dgetrf_seq );
    /* printf( "DGEMM TILED OPENMP:\n" );
    testall_dgemm_tiled( (dgemm_tiled_fct_t) tested_dgemm );
    printf( "DGEMM TILED SEQUENTIAL:\n" );
    testall_dgetrf_tiled( (dgetrf_tiled_fct_t) tested_dgetrf ); */

    return EXIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif

