#ifndef MY_LAPACK_H
#define MY_LAPACK_H

#define BLOCK_SIZE 128

namespace my_lapack {
    enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
    enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
    enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
    enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
    enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

    double my_ddot( const int N, const double *X, const int incX, const double *Y, const int incY );

    void my_daxpy( const int N, const double alpha, const double *X, const int incX, double *Y, const int incY );

    void my_dgemv( CBLAS_ORDER    layout,
                   CBLAS_TRANSPOSE TransA,
                   int             M,
                   int             N,
                   double          alpha,
                   const double *  A,
                   int             lda,
                   const double *  X,
                   int             incX,
                   double          beta,
                   double *        Y,
                   const int       incY );

    void my_dgemm_scalaire( const enum CBLAS_ORDER     Order,
                            const enum CBLAS_TRANSPOSE TransA,
                            const enum CBLAS_TRANSPOSE TransB,
                            const int                  M,
                            const int                  N,
                            const int                  K,
                            const double               alpha,
                            const double *             A,
                            const int                  lda,
                            const double *             B,
                            const int                  ldb,
                            const double               beta,
                            double *                   C,
                            const int                  ldc );
                            
    void my_dger( const enum CBLAS_ORDER order,
                  const int              M,
                  const int              N,
                  const double           alpha,
                  const double *         X,
                  const int              incX,
                  const double *         Y,
                  const int              incY,
                  double *               A,
                  const int              lda);

    void my_dgemm( const enum CBLAS_ORDER     Order,
                   const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_TRANSPOSE TransB,
                   const int                  M,
                   const int                  N,
                   const int                  K,
                   const double               alpha,
                   const double *             A,
                   const int                  lda,
                   const double *             B,
                   const int                  ldb,
                   const double               beta,
                   double *                   C,
                   const int                  ldc );

    /*
    int my_dgetrf( const enum CBLAS_ORDER Order, int m, int n, double *a, int lda, int *ipiv );
    */

} // namespace my_lapack
#endif
