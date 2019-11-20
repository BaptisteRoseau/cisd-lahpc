#include "Mat.h"
#include "my_lapack.h"
#include "util.h"

#include <chrono>
#include <fstream>
#include <iostream>

template<typename T, size_t N>
char ( &ArraySizeHelper( T ( & )[N] ) )[N];

#define ARRAY_SIZE( A ) ( sizeof( ArraySizeHelper( A ) ) )


using namespace my_lapack;
using namespace std;

/*============ UTILS FOR TESTING PURPOSE =============== */

void print_test_result(int result, int *nb_success, int *nb_tests)
{
    if (result == EXIT_SUCCESS) {
        printf("\x1B[32mSUCCESS\x1B[0m\n");
        (*nb_success)++;
    } else {

        printf("\x1B[31mFAILED\x1B[0m\n");
    }

    (*nb_tests)++;
}

void print_test_summary(int nb_success, int nb_tests)
{
    if (nb_success == nb_tests)
        printf("TESTS SUMMARY: \t\x1B[32m%d\x1B[0m/%d\n", nb_success, nb_tests);
    else
        printf("TESTS SUMMARY: \t\x1B[31m%d\x1B[0m/%d\n", nb_success, nb_tests);
}

/*============ TESTS DEFINITION =============== */

int test_perf_dgemm()
{
    printf("%s ", __func__);

    fstream fout;
    fout.open( "test_perf_dgemm.csv", ios::out | ios::app );
    size_t                        powInc = 2;
    size_t                        len    = 2;
    size_t                        lenMax = powInc << 11;
    double                        valMax = 200;
    double                        alpha  = 0.75;
    double                        beta   = 0;
    Mat *                         m1, *m2, *m3;
    std::chrono::duration<double> diff;

    while ( len < lenMax ) {
        m1      = new Mat( len, len, 10 );
        m2      = new Mat( len, len, 10 );
        m3      = new Mat( len, len, 10 );
        auto t0 = chrono::system_clock::now();
        my_dgemm( CblasColMajor,
                  CblasNoTrans,
                  CblasNoTrans,
                  m1->dimX(),
                  m2->dimY(),
                  m1->dimY(),
                  alpha,
                  m1->get(),
                  m1->dimX(),
                  m2->get(),
                  m2->dimX(),
                  beta,
                  m3->get(),
                  m3->dimX() );
        auto t1 = chrono::system_clock::now();
        diff    = t1 - t0;

        // Insert the data to file
        fout << len << ", " << ( t1 - t0 ).count() << "\n";
        cout << "Len: " << len << "\tTime: " << ( diff ).count() << endl;

        delete m1;
        delete m2;
        delete m3;
        len *= powInc;
    }

    fout.close();

    return EXIT_SUCCESS;
}

/*============ MAIN CALL =============== */

int main( int argc, char **argv )
{
    printf("----------- TEST PERF -----------\n");

    int nb_success = 0;
    int nb_tests = 0;

    print_test_result(test_perf_dgemm(), &nb_success, &nb_tests);

    print_test_summary(nb_success, nb_tests);

    return EXIT_SUCCESS;
}
