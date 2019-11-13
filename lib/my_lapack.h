#ifndef MY_LAPACK_H
#define MY_LAPACK_H

namespace my_lapack {

    enum class LAHPC_LAYOUT { ColMajor = 0, RowMajor = 1 };
    enum class LAHPC_TRANSPOSE { NoTrans = 0, Trans = 1 };
    enum class LAHPC_ORDER {CblasRowMajor=0, CblasColMajor=1};

    double my_ddot( const int N, const double *X, const int incX, const double *Y, const int incY );

    void my_daxpy( const int N, const double alpha, const double *X, const int incX, double *Y, const int incY );

    void my_dgemv( LAHPC_LAYOUT    layout,
                   LAHPC_TRANSPOSE TransA,
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

    void my_dgemm_scalaire( const enum LAHPC_ORDER     Order,
                            const enum LAHPC_TRANSPOSE TransA,
                            const enum LAHPC_TRANSPOSE TransB,
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
                            
    void my_dger( const enum LAHPC_ORDER order,
                  const int              M,
                  const int              N,
                  const double           alpha,
                  const double *         X,
                  const int              incX,
                  const double *         Y,
                  const int              incY,
                  double *               A,
                  const int              lda)

    void my_dgemm( const enum LAHPC_ORDER     Order,
                   const enum LAHPC_TRANSPOSE TransA,
                   const enum LAHPC_TRANSPOSE TransB,
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
    int my_dgetrf( const enum LAHPC_ORDER Order, int m, int n, double *a, int lda, int *ipiv );
    */

} // namespace my_lapack
#endif
