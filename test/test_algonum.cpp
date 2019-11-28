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
    testall_dgemm( (dgemm_fct_t) my_dgemm_openmp );
    //testall_dgemm( (dgemm_fct_t) my_dgemm_seq );
    //testall_dgetrf( (dgemm_tiled_fct_t) my_dgetrf_openmp );
    //testall_dgetrf( (dgemm_tiled_fct_t) my_dgetrf_seq );
    //testall_dgemm_tiled( (dgemm_tiled_fct_t) tested_dgemm );
    //testall_dgetrf_tiled( (dgetrf_tiled_fct_t) tested_dgetrf );

    return EXIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif

