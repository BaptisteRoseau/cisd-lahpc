#ifndef MY_LAPACK_H
#define MY_LAPACK_H

#include <cblas.h>
//#include <flops.h>
//#include <lapacke.h> //header include error
//#include <perf.h>

/*===================== Functions requiered for the test and librairies */

double my_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);

void my_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
void my_dgemm_scalaire(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
int my_dgetrf(const enum CBLAS_ORDER Order, int m, int n, double* a, int lda, int* ipiv);


#endif