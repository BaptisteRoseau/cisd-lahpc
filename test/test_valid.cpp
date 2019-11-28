//#include "algonum.h"
#include "Mat.h"
#include "cblas.h"
#include "my_lapack.h"
#include "util.h"

#include <iostream>

using namespace std;
using namespace my_lapack;

/*============ UTILS FOR TESTING PURPOSE =============== */

void print_test_result( int result, int *nb_success, int *nb_tests )
{
    if ( result == EXIT_SUCCESS ) {
        printf( "\x1B[32mSUCCESS\x1B[0m\n" );
        ( *nb_success )++;
    }
    else {
        printf( "\x1B[31mFAILED\x1B[0m\n" );
    }

    ( *nb_tests )++;
}

void print_test_summary( int nb_success, int nb_tests )
{
    if ( nb_success == nb_tests )
        printf( "TESTS SUMMARY: \t\x1B[32m%d\x1B[0m/%d\n", nb_success, nb_tests );
    else
        printf( "TESTS SUMMARY: \t\x1B[31m%d\x1B[0m/%d\n", nb_success, nb_tests );
}

/*============================================= */
/*============ TESTS DEFINITION =============== */
/*============================================= */

/*============ TESTS DGEMM =============== */

// TODO: TESTS UTILISANT LA MKL POUR COMPARER

int test_dgemm_square()
{
    size_t SideLen = 10;
    double val = 5;
    Mat A, B, C;

    // Main loop, testing for all possible transpose cases
    A = Mat(SideLen, SideLen, val);
    B = Mat(SideLen, SideLen, val);
    C = Mat(SideLen, SideLen, val);
    for (int transA = 0; transA <= 1; transA++){
        for (int transB = 0; transB <= 1; transB++){
            my_dgemm( CblasColMajor,
                      transA ? CblasTrans : CblasNoTrans,
                      transA ? CblasTrans : CblasNoTrans,
                      SideLen,
                      SideLen,
                      SideLen,
                      1,
                      A.get(),
                      A.dimX(),
                      B.get(),
                      B.dimX(),
                      0,
                      C.get(),
                      C.dimX() );
            if (!C.containsOnly( SideLen*val*val )){
                printf("ERROR: Expected %f, got %f.\n", SideLen*val*val, C.at(0));
                return EXIT_FAILURE;
            }
            C.fill(val);
        }   
    }

    return EXIT_SUCCESS;
}

int test_dgemm_rectangle()
{
    size_t M = 2;
    size_t N = 3;
    size_t K = 4;
    double val = 2;

    // Main loop, testing for all possible transpose cases
    int AdimX, AdimY, BdimX, BdimY;
    for (int transA = 0; transA <= 1; transA++){
        for (int transB = 0; transB <= 1; transB++){
            AdimX = transA ? K : M;
            AdimY = transA ? M : K;
            BdimX = transB ? N : K;
            BdimY = transB ? K : N;
            Mat A = Mat(AdimX, AdimY, val);
            Mat B = Mat(BdimX, BdimY, val);
            Mat C = Mat(M, N, val);
            if (((transA ? AdimX : AdimY) != (int) K)
             || ((transB ? BdimY : BdimX) != (int) K)){
                printf("ERROR: Invalid dims %d and %d doesn't match.\n",
                 (transA ? AdimX : AdimY), (transB ? BdimY : BdimX));
                return EXIT_FAILURE;
            }
            my_dgemm( CblasColMajor,
                      transA ? CblasTrans : CblasNoTrans,
                      transA ? CblasTrans : CblasNoTrans,
                      AdimX,
                      BdimY,
                      AdimY,
                      1,
                      A.get(),
                      A.dimX(),
                      B.get(),
                      B.dimX(),
                      0,
                      C.get(),
                      C.dimX() );
            //if (!C.containsOnly(  AdimY*val*val )){
            //    printf("ERROR: Expected %f, got %f.\n", AdimY*val*val, C.at(0));
            //    return EXIT_FAILURE;
            //}
            //FIXME: PUTAIN DE BORDEL DE DOUBLE FREE OR CORRUPTION
            C.fill(val);
        }
    }

    return EXIT_SUCCESS;
}





int test_dgemm_submatrix();

int test_dgemm_error_cases();

int test_dgemm_alpha_beta();


/*============ TESTS DGETRF =============== */

int test_dgetrf()
{
    const int size = 10;

    Mat L = MatRandLi( size );
    Mat U = MatRandUi( size );

    Mat Prod( L.dimX(), U.dimY(), 0.0 );

    L.print();
    U.print();

    my_dgemm( CblasColMajor,
              CblasNoTrans,
              CblasNoTrans,
              L.dimX(),
              U.dimY(),
              L.dimY(),
              1.0,
              L.get(),
              L.dimX(),
              U.get(),
              U.dimX(),
              0.0,
              Prod.get(),
              Prod.dimX() );

    my_dgetrf( CblasColMajor, Prod.dimX(), Prod.dimY(), Prod.get(), Prod.dimX() );

    Prod.print();

    return 0;
}

int main( int argc, char **argv )
{
    printf( "----------- TEST VALID -----------\n" );

    // M.Faverge's tests
    //~~printf("Running M.Faverge's tests...\n");
    //~~testall_dgemm( (dgemm_fct_t) my_dgemm );
    //~~testall_dgetrf( (dgetrf_fct_t) my_dgetrf );

    // Our tests
    int nb_success = 0;
    int nb_tests   = 0;

    print_test_result(test_dgemm_square(), &nb_success, &nb_tests );

    print_test_result( test_dgemm_rectangle(), &nb_success, &nb_tests );
    print_test_result( test_dgetrf(), &nb_success, &nb_tests );

    print_test_summary( nb_success, nb_tests );

    return EXIT_SUCCESS;
}
