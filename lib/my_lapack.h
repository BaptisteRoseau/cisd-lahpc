#ifndef MY_LAPACK_H
#define MY_LAPACK_H

namespace my_lapack {

    enum CBLAS_LAYOUT { CblasRowMajor = 101, CblasColMajor = 102 };
    enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
    enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
    enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };
    enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };

    double my_ddot( const int N, const double *X, const int incX, const double *Y, const int incY );

    void my_daxpy( const int N, const double alpha, const double *X, const int incX, double *Y, const int incY );

    void my_dgemv( enum CBLAS_LAYOUT    layout,
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

    void my_dgemm_scalaire( enum CBLAS_LAYOUT    Order,
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

    void my_dger( enum CBLAS_LAYOUT layout,
                  int               M,
                  int               N,
                  double            alpha,
                  const double *    X,
                  int               incX,
                  const double *    Y,
                  int               incY,
                  double *          A,
                  int               lda );

    void my_dgemm( enum CBLAS_LAYOUT    Order,
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

} // namespace my_lapack
#endif
