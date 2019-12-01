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

int test_dgemm_square()
{
    printf( "%s:\t", __func__ );

    size_t SideLen = 260;
    double alpha, beta, val = (double)rand() / RAND_MAX;
    Mat    A, B, C;

    // Main loop, testing for all possible transpose cases
    A = Mat( SideLen, SideLen, val );
    B = Mat( SideLen, SideLen, val );
    C = Mat( SideLen, SideLen, val );
    for ( int transA = 0; transA <= 1; transA++ ) {
        for ( int transB = 0; transB <= 1; transB++ ) {
            alpha = (double)rand() / RAND_MAX;
            beta  = (double)rand() / RAND_MAX;
            my_dgemm( CblasColMajor,
                      transA ? CblasTrans : CblasNoTrans,
                      transA ? CblasTrans : CblasNoTrans,
                      SideLen,
                      SideLen,
                      SideLen,
                      alpha,
                      A.get(),
                      A.dimX(),
                      B.get(),
                      B.dimX(),
                      beta,
                      C.get(),
                      C.dimX() );
            if ( !C.containsOnly( SideLen * val * val * alpha + val * beta ) ) {
                printf( "ERROR: Expected %f, got %f.\t", SideLen * val * val * alpha + val * beta, C.at( 0 ) );
                return EXIT_FAILURE;
            }
            C.fill( val );
        }
    }

    return EXIT_SUCCESS;
}

// FIXME: This test is bugged.
int test_dgemm_rectangle()
{
    printf( "%s:\t", __func__ );

    size_t M = 200;
    size_t N = 100;
    size_t K = 260;
    double alpha, beta, val;
    Mat    A = Mat( M, K );
    Mat    B = Mat( K, N );
    Mat    C = Mat( M, N );

    // Main loop, testing for all possible transpose cases
    int AdimX, AdimY, BdimX, BdimY;
    for ( int transA = 0; transA <= 1; transA++ ) {
        for ( int transB = 0; transB <= 1; transB++ ) {
            alpha = (double)rand() / RAND_MAX;
            beta  = (double)rand() / RAND_MAX;
            val   = (double)rand() / RAND_MAX;
            AdimX = transA ? K : M;
            AdimY = transA ? M : K;
            BdimX = transB ? N : K;
            BdimY = transB ? K : N;
            // A.reshape(AdimX, AdimY, val);
            // B.reshape(BdimX, BdimY, val);
            C.fill( val );
            if ( ( ( transA ? AdimX : AdimY ) != (int)K ) || ( ( transB ? BdimY : BdimX ) != (int)K ) ) {
                printf( "ERROR: Invalid dims %d and %d doesn't match.\t",
                        ( transA ? AdimX : AdimY ),
                        ( transB ? BdimY : BdimX ) );
                return EXIT_FAILURE;
            }
            my_dgemm( CblasColMajor,
                      transA ? CblasTrans : CblasNoTrans,
                      transB ? CblasTrans : CblasNoTrans,
                      AdimX,
                      BdimY,
                      AdimY,
                      alpha,
                      A.get(),
                      AdimX,
                      B.get(),
                      BdimX,
                      beta,
                      C.get(),
                      C.dimX() );
            if ( !C.containsOnly( AdimY * val * val * alpha + val * beta ) ) {
                printf( "ERROR: Expected %f, got %f.\t", AdimY * val * val * alpha + val * beta, C.at( 0 ) );
                return EXIT_FAILURE;
            }
        }
    }

    return EXIT_SUCCESS;
}

int test_dgemm_submatrix();

int test_dgemm_error_cases();

/*============ TESTS DGETRF =============== */

int test_dgetrf()
{
    printf("%s:\t", __func__);

    const int size = 20;

    Mat L = MatRandLi( size );
    Mat U = MatRandUi( size );

    Mat LU( L.numRow(), L.numCol() );
    for ( int i = 0; i < L.numRow(); ++i ) {
        for ( int j = 0; j < i; ++j ) {
            LU.at( i, j ) = L.at( i, j );
        }
        for ( int j = i; j < L.numCol(); ++j ) {
            LU.at( i, j ) = U.at( i, j );
        }
    }

    Mat Prod( L.dimX(), U.dimY(), 0.0 );

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

   /* LU.print();
    Prod.print();*/
    bool equal = LU.equals(Prod, 1);

    return !equal;
}

int main( int argc, char **argv )
{
    printf( "----------- TEST VALID -----------\n" );

    int nb_success = 0;
    int nb_tests   = 0;
    // srand(time(NULL));

    print_test_result( test_dgemm_square(), &nb_success, &nb_tests );
    // print_test_result( test_dgemm_rectangle(), &nb_success, &nb_tests );
    print_test_result( test_dgetrf(), &nb_success, &nb_tests );

    print_test_summary( nb_success, nb_tests );

    return EXIT_SUCCESS;
}
