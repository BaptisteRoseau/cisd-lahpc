#ifndef MY_LAPACK_H
#define MY_LAPACK_H

#include "cblas.h"

namespace my_lapack {

    double my_ddot( const int N, const double *X, const int incX, const double *Y, const int incY );

    void my_daxpy( const int N, const double alpha, const double *X, const int incX, double *Y, const int incY );

    void my_dgemv( enum CBLAS_ORDER     layout,
                   enum CBLAS_TRANSPOSE TransA,
                   int                  M,
                   int                  N,
                   double               alpha,
                   const double *       A,
                   int                  lda,
                   const double *       X,
                   int                  incX,
                   double               beta,
                   double *             Y,
                   const int            incY );

    void my_dgemm_scalaire( enum CBLAS_ORDER     layout,
                            enum CBLAS_TRANSPOSE TransA,
                            enum CBLAS_TRANSPOSE TransB,
                            int                  M,
                            int                  N,
                            int                  K,
                            double               alpha,
                            const double *       A,
                            int                  lda,
                            const double *       B,
                            int                  ldb,
                            double               beta,
                            double *             C,
                            int                  ldc );

    void my_dger( enum CBLAS_ORDER layout,
                  int              M,
                  int              N,
                  double           alpha,
                  const double *   X,
                  int              incX,
                  const double *   Y,
                  int              incY,
                  double *         A,
                  int              lda );

    void my_dgemm( enum CBLAS_ORDER     Order,
                   enum CBLAS_TRANSPOSE TransA,
                   enum CBLAS_TRANSPOSE TransB,
                   int                  M,
                   int                  N,
                   int                  K,
                   double               alpha,
                   const double *       A,
                   int                  lda,
                   const double *       B,
                   int                  ldb,
                   double               beta,
                   double *             C,
                   int                  ldc );

    //************************************
    // Method:    my_dgetrf2
    // FullName:  my_lapack::my_dgetrf2
    // Access:    public
    // Returns:   void
    // Parameter: int M - Number of rows of matrix A
    // Parameter: int N - Number of columns of matrix A
    // Parameter: double * A - Dim (lda, N). Upon exit, holds L, U such that A = P * L * U
    // Parameter: int lda - Leading dimension of matrix A
    // Parameter: int * ipiv - Dim (min(M,N)). Pivot indices, row i of A was interchanged with ipiv[i]
    // Parameter: int * info - 0 if success, if info > 0 then U(info - 1,info - 1) == 0.0
    //************************************
    void my_dgetf2( int M, int N, double *A, int lda, int *ipiv, int *info );

    void my_dtrsm( char          side,
                   char          uplo,
                   char          transA,
                   char          diag,
                   int           M,
                   int           N,
                   double        alpha,
                   const double *A,
                   int           lda,
                   double *      B,
                   int           ldb );

    int my_idamax( int N, double *dx, int incX );

    void my_dscal( int N, double da, double *dx, int incX );

    void my_dlaswp( int N, double *A, int lda, int k1, int k2, int *ipv, int incX );

} // namespace my_lapack
#endif
