#include "Mat.h"
#include "my_lapack.h"
#include "algonum.h"
#include "util.h"

#include <chrono>
#include <fstream>
#include <iostream>

//TODO: Calculer la perf théorique

template<typename T, size_t N>
char ( &ArraySizeHelper( T ( & )[N] ) )[N];

#define ARRAY_SIZE( A ) ( sizeof( ArraySizeHelper( A ) ) )

using namespace my_lapack;
using namespace std;

/*============ TESTS DEFINITION =============== */

double flops_dgemm(size_t M, size_t N, size_t K, double beta, double seconds){
    return (2*M*N*K)/seconds;
}

int test_perf_dgemm(dgemm_fct_t dgemm_func, 
                    const char *csv_file_time,
                    const char *csv_file_flops,
                    const char *curve_title,
                    bool appendToFile)
{
    printf("%s, curve \"%s\" into \"%s\" (time) and \"%s\" (flops)\n",
     __func__, curve_title, csv_file_time, csv_file_flops);

    // Output stream
    fstream fout_time;
    fstream fout_flops;
    fout_time.open( csv_file_time, ios::out | (appendToFile ? ios::app : ios::trunc ) );
    fout_flops.open( csv_file_flops, ios::out | (appendToFile ? ios::app : ios::trunc ) );
    if (!appendToFile){
        // WARNING: Spaces between comma aren't allowed !
        // Title, XTitle, YTitle, XScale, YScale
        fout_time << "DGEMM time taken,Matrix dimension,Time (s),linear,log\n";
        fout_flops << "DGEMM GFlops,Matrix dimension,GFlops,linear,log\n";
    }
    fout_time  << curve_title << endl;
    fout_flops << curve_title << endl;

    // Test variables
    size_t powInc    = 2;
    size_t len       = 256;
    size_t lenMax    = powInc << 11;
    double alpha     = 0.75;
    double beta      = 2;
    Mat *  m1, *m2, *m3;
    chrono::duration<double> diff;
    double Gflops, Gflops_mean;
    int counter = 0;

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
        Gflops = flops_dgemm(len, len, len, beta, ( diff ).count())/1e9;
        Gflops_mean += Gflops;
        fout_time << len << ", " << ( diff ).count() << "\n";
        cout << "Len: " << len << "\tTime: " << ( diff ).count() << "\tGFlop/s: " << Gflops << endl;

        len *= powInc;
        counter++;

        // Freeing memory
        delete m1;
        delete m2;
        delete m3;
    }

    // Writing Gflops into file
    Gflops_mean /= counter;
    fout_flops << len << ", " << Gflops_mean << "\n";

    cout << "Done.\n";
    fout_time.close();
    fout_flops.close();

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

void print_usage(){
    cerr << "Usage: test_perf <output file time> <output file flops>\n";
}

int main( int argc, char **argv )
{
    printf("----------- TEST PERF -----------\n");
    if (argc < 3){
        print_usage();
        return EXIT_FAILURE;
    }
    test_perf_dgemm(my_dgemm_scal_seq, argv[1], argv[2], "Sequential Scalar", false);
    test_perf_dgemm(my_dgemm_scal_openmp, argv[1], argv[2], "OpenMP Scalar", true);
    test_perf_dgemm(my_dgemm_seq, argv[1], argv[2], "Sequential", true);
    test_perf_dgemm(my_dgemm_openmp, argv[1], argv[2], "OpenMP", true);
    
    return EXIT_SUCCESS;
}
