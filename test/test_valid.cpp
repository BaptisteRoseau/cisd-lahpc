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
    printf( "%s ", __func__ );

    Mat *m1     = new Mat( 10, 10, 1 );
    Mat *m2     = new Mat( 10, 10, 1 );
    Mat *m3     = new Mat( 10, 10, 1 );
    Mat *m_my   = new Mat( 10, 10, 0 );
    Mat *m_blas = new Mat( 10, 10, 0 );

    // M1, M2
    my_dgemm( CblasColMajor,
              CblasNoTrans,
              CblasNoTrans,
              m1->dimX(),
              m2->dimY(),
              m1->dimY(),
              2,
              m1->get(),
              m1->dimX(),
              m2->get(),
              m2->dimX(),
              3,
              m_my->get(),
              m_my->dimX() );
    cblas_dgemm( CblasColMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 m1->dimX(),
                 m2->dimY(),
                 m1->dimY(),
                 2,
                 m1->get(),
                 m1->dimX(),
                 m2->get(),
                 m2->dimX(),
                 3,
                 m_blas->get(),
                 m_blas->dimX() );

    m_my->print();
    m_blas->print();
    if ( !m_my->equals( *m_blas ) ) { return EXIT_FAILURE; }

    // M1, TM2
    m3->fill( 0 );
    my_dgemm( CblasColMajor,
              CblasNoTrans,
              CblasTrans,
              m1->dimX(),
              m1->dimY(),
              m2->dimX(),
              1,
              m1->get(),
              m1->dimX(),
              m2->get(),
              m2->dimX(),
              0,
              m3->get(),
              m3->dimX() );

    if ( !m3->containsOnly( 10 ) ) { return EXIT_FAILURE; }

    // TM1, M2
    m3->fill( 0 );
    my_dgemm( CblasColMajor,
              CblasTrans,
              CblasNoTrans,
              m1->dimY(),
              m1->dimX(),
              m2->dimY(),
              1,
              m1->get(),
              m1->dimX(),
              m2->get(),
              m2->dimX(),
              0,
              m3->get(),
              m3->dimX() );

    if ( !m3->containsOnly( 10 ) ) { return EXIT_FAILURE; }

    // TM1, TM2
    m3->fill( 0 );
    my_dgemm( CblasColMajor,
              CblasTrans,
              CblasTrans,
              m1->dimY(),
              m1->dimX(),
              m2->dimX(),
              1,
              m1->get(),
              m1->dimX(),
              m2->get(),
              m2->dimX(),
              0,
              m3->get(),
              m3->dimX() );

    if ( !m3->containsOnly( 10 ) ) { return EXIT_FAILURE; }

    delete m1;
    delete m2;
    delete m3;

    return EXIT_SUCCESS;
}

int test_dgemm_rectangle()
{
    printf( "%s ", __func__ );

    Mat *m1 = new Mat( 10, 20, 1 );
    Mat *m2 = new Mat( 20, 10, 1 );
    Mat *m3 = new Mat( 10, 10, 0 );

    // M1, M2
    my_dgemm( CblasColMajor,
              CblasNoTrans,
              CblasNoTrans,
              m1->dimX(),
              m2->dimY(),
              m1->dimY(),
              1,
              m1->get(),
              m1->dimX(),
              m2->get(),
              m2->dimX(),
              0,
              m3->get(),
              m3->dimX() );

    if ( !m3->containsOnly( 20 ) ) {
        m3->print();
        delete m1;
        delete m2;
        delete m3;

        return EXIT_FAILURE;
    }

    // M1, TM2
    // delete m1;
    // delete m2;
    // delete m3;
    m1 = new Mat( 10, 20, 1 );
    m2 = new Mat( 10, 20, 1 );
    m3 = new Mat( 10, 10, 0 );
    my_dgemm( CblasColMajor,
              CblasNoTrans,
              CblasTrans,
              m1->dimX(),
              m1->dimY(),
              m2->dimX(),
              1,
              m1->get(),
              m1->dimX(),
              m2->get(),
              m2->dimX(),
              0,
              m3->get(),
              m3->dimX() );
    m3->print();
    if ( !m3->containsOnly( 20 ) ) {
        delete m1;
        delete m2;
        delete m3;

        return EXIT_FAILURE;
    }

    // TM1, M2
    // delete m1;
    // delete m2;
    // delete m3;
    m1 = new Mat( 20, 10, 1 );
    m2 = new Mat( 20, 10, 1 );
    m3 = new Mat( 10, 10, 0 );
    my_dgemm( CblasColMajor,
              CblasTrans,
              CblasNoTrans,
              m1->dimY(),
              m1->dimX(),
              m2->dimY(),
              1,
              m1->get(),
              m1->dimX(),
              m2->get(),
              m2->dimX(),
              0,
              m3->get(),
              m3->dimX() );
    m3->print();
    if ( !m3->containsOnly( 20 ) ) {
        delete m1;
        delete m2;
        delete m3;

        return EXIT_FAILURE;
    }

    // TM1, TM2
    // delete m1;
    // delete m2;
    // delete m3;
    m1 = new Mat( 20, 10, 1 );
    m2 = new Mat( 10, 20, 1 );
    m3 = new Mat( 10, 10, 0 );
    my_dgemm( CblasColMajor,
              CblasTrans,
              CblasTrans,
              m1->dimY(),
              m1->dimX(),
              m2->dimX(),
              1,
              m1->get(),
              m1->dimX(),
              m2->get(),
              m2->dimX(),
              0,
              m3->get(),
              m3->dimX() );
    m3->print();
    if ( !m3->containsOnly( 20 ) ) {
        delete m1;
        delete m2;
        delete m3;

        return EXIT_FAILURE;
    }

    delete m1;
    delete m2;
    delete m3;

    return EXIT_SUCCESS;
}

int test_dgemm_submatrix();

int test_dgemm_error_cases();

int test_dgemm_alpha_beta();

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

    print_test_result( test_dgemm_square(), &nb_success, &nb_tests );
    print_test_result( test_dgemm_rectangle(), &nb_success, &nb_tests );

    test_dgetrf();

    print_test_summary( nb_success, nb_tests );

    return EXIT_SUCCESS;
}
