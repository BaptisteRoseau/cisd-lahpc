#include "Mat.h"
#include "my_lapack.h"
#include "algonum.h"
#include "util.h"

#include <chrono>
#include <fstream>
#include <iostream>

template<typename T, size_t N>
char ( &ArraySizeHelper( T ( & )[N] ) )[N];

#define ARRAY_SIZE( A ) ( sizeof( ArraySizeHelper( A ) ) )

using namespace my_lapack;
using namespace std;

/*============ TESTS DEFINITION =============== */

int test_perf_dgemm(dgemm_fct_t dgemm_func, 
                    const char *csv_file, 
                    const char *curve_title,
                    bool appendToFile)
{
    printf("%s, curve %s into %s\n", __func__, curve_title, csv_file);

    // Output stream
    fstream fout;
    fout.open( csv_file, ios::out | (appendToFile ? ios::app : ios::trunc ) );
    if (!appendToFile){
        // WARNING: Spaces between comma aren't allowed !
        // Title, XTitle, YTitle, XScale, YScale
        fout << "DGEMM time taken,Matrix dimension,Time (s),linear,log\n";
    }
    fout << curve_title << endl;

    // Test variables
    size_t powInc    = 2;
    size_t len       = 16;
    size_t lenMax    = powInc << 11;
    double alpha     = 0.75;
    double beta      = 2;
    Mat *  m1, *m2, *m3;
    chrono::duration<double> diff;

    // Main loop (exponential increment)
    while ( len < lenMax ) {
        m1      = new Mat( len, len, 10 );
        m2      = new Mat( len, len, 10 );
        m3      = new Mat( len, len, 10 );
        auto t0 = chrono::system_clock::now();
        dgemm_func( CblasColMajor,
                    CblasTrans,
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
        fout << len << ", " << ( diff ).count() << "\n";
        cout << "Len: " << len << "\tTime: " << ( diff ).count() << endl;

        // Freeing memory
        delete m1;
        delete m2;
        delete m3;
        len *= powInc;
    }

    cout << "Done.\n";
    fout.close();

    return EXIT_SUCCESS;
}

/*============ MAIN CALL =============== */

/* 
CSV OUTPUT FORMAT:
TITLE, XTITLE, YTITLE, XSCALE ("linear" or "log"), YSCALE
CURVE LABEL
x0_0, y0_0
x0_1, y0_1
...
CURVE LABEL
x1_0, y1_0
x1_1, y1_1
x1_2, y1_2
...
*/

int main( int argc, char **argv )
{
    printf("----------- TEST PERF -----------\n");

    test_perf_dgemm(my_dgemm_scal_seq, "dgemm.csv", "Sequential Scalar", false);
    test_perf_dgemm(my_dgemm_scal_openmp, "dgemm.csv", "OpenMP Scalar", true);
    test_perf_dgemm(my_dgemm_seq, "dgemm.csv", "Sequential", true);
    test_perf_dgemm(my_dgemm_openmp, "dgemm.csv", "OpenMP", true);
    
    return EXIT_SUCCESS;
}
