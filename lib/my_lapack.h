#ifndef MY_LAPACK_H
#define MY_LAPACK_H

#include "cblas.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace my_lapack {

    double my_ddot_seq( const int N, const double *X, const int incX, const double *Y, const int incY );
    double my_ddot_openmp( const int N, const double *X, const int incX, const double *Y, const int incY );

    void my_daxpy_seq( const int N, const double alpha, const double *X, const int incX, double *Y, const int incY );
    void my_daxpy_openmp( const int N, const double alpha, const double *X, const int incX, double *Y, const int incY );

    void my_dgemv_seq( CBLAS_ORDER     layout,
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
    void my_dgemv_openmp( CBLAS_ORDER     layout,
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

    void my_dgemm_scal_seq( CBLAS_ORDER     layout,
                            CBLAS_TRANSPOSE TransA,
                            CBLAS_TRANSPOSE TransB,
                            int             M,
                            int             N,
                            int             K,
                            double          alpha,
                            const double *  A,
                            int             lda,
                            const double *  B,
                            int             ldb,
                            double          beta,
                            double *        C,
                            int             ldc );
    void my_dgemm_scal_openmp( CBLAS_ORDER     layout,
                               CBLAS_TRANSPOSE TransA,
                               CBLAS_TRANSPOSE TransB,
                               int             M,
                               int             N,
                               int             K,
                               double          alpha,
                               const double *  A,
                               int             lda,
                               const double *  B,
                               int             ldb,
                               double          beta,
                               double *        C,
                               int             ldc );

    void my_dger_seq( CBLAS_ORDER   layout,
                      int           M,
                      int           N,
                      double        alpha,
                      const double *X,
                      int           incX,
                      const double *Y,
                      int           incY,
                      double *      A,
                      int           lda );
    void my_dger_openmp( CBLAS_ORDER   layout,
                         int           M,
                         int           N,
                         double        alpha,
                         const double *X,
                         int           incX,
                         const double *Y,
                         int           incY,
                         double *      A,
                         int           lda );

    void my_dgemm_seq( CBLAS_ORDER     Order,
                       CBLAS_TRANSPOSE TransA,
                       CBLAS_TRANSPOSE TransB,
                       int             M,
                       int             N,
                       int             K,
                       double          alpha,
                       const double *  A,
                       int             lda,
                       const double *  B,
                       int             ldb,
                       double          beta,
                       double *        C,
                       int             ldc );
    void my_dgemm_openmp( CBLAS_ORDER     Order,
                          CBLAS_TRANSPOSE TransA,
                          CBLAS_TRANSPOSE TransB,
                          int             M,
                          int             N,
                          int             K,
                          double          alpha,
                          const double *  A,
                          int             lda,
                          const double *  B,
                          int             ldb,
                          double          beta,
                          double *        C,
                          int             ldc );

    void my_dgemm_mpi( CBLAS_ORDER Order,
                          CBLAS_TRANSPOSE TransA,
                          CBLAS_TRANSPOSE TransB,
                          int M,
                          int N,
                          int K,
                          double alpha,
                          const double * a,
                          int lda,
                          const double * b,
                          int ldb,
                          double beta,
                          double * c,
                          int ldc );

    void my_dgetf2_seq( CBLAS_ORDER order, int M, int N, double *A, int lda );
    void my_dgetf2_openmp( CBLAS_ORDER order, int M, int N, double *A, int lda );

    void my_dtrsm_seq( CBLAS_ORDER     layout,
                       CBLAS_SIDE      Side,
                       CBLAS_UPLO      Uplo,
                       CBLAS_TRANSPOSE transA,
                       CBLAS_DIAG      Diag,
                       int             M,
                       int             N,
                       double          alpha,
                       const double *  A,
                       int             lda,
                       double *        B,
                       int             ldb );
    void my_dtrsm_openmp( CBLAS_ORDER     layout,
                          CBLAS_SIDE      Side,
                          CBLAS_UPLO      Uplo,
                          CBLAS_TRANSPOSE transA,
                          CBLAS_DIAG      Diag,
                          int             M,
                          int             N,
                          double          alpha,
                          const double *  A,
                          int             lda,
                          double *        B,
                          int             ldb );

    void my_dgetrf_seq( CBLAS_ORDER order, int M, int N, double *A, int lda );
    void my_dgetrf_openmp( CBLAS_ORDER order, int M, int N, double *A, int lda );

    int my_idamax_seq( int N, double *dx, int incX );
    int my_idamax_openmp( int N, double *dx, int incX );

    void my_dscal_seq( int N, double da, double *dx, int incX );
    void my_dscal_openmp( int N, double da, double *dx, int incX );

    void my_dlaswp_seq( int N, double *A, int lda, int k1, int k2, int *ipv, int incX );
    void my_dlaswp_openmp( int N, double *A, int lda, int k1, int k2, int *ipv, int incX );

    void my_dlacpy( int M, int N, const double *a, int lda, double *b, int ldb );


// Macro definitions to respect our previous naming
#ifdef _my_lapack_seq
    #define my_ddot my_ddot_seq
    #define my_daxpy my_daxpy_seq
    #define my_dgemv my_dgemv_seq
    #define my_dgemm_scal my_dgemm_scal_seq
    #define my_dger my_dger_seq
    #define my_dgemm my_dgemm_seq
    #define my_dgetf2 my_dgetf2_seq
    #define my_dgetrf my_dgetrf_seq
    #define my_dtrsm my_dtrsm_seq
    #define my_idamax my_idamax_seq
    #define my_dscal my_dscal_seq
    #define my_dlaswp my_dlaswp_seq
#else
    #if defined _my_lapack_omp || defined _my_lapack_all
        #define my_ddot my_ddot_openmp
        #define my_daxpy my_daxpy_openmp
        #define my_dgemv my_dgemv_openmp
        #define my_dgemm_scal my_dgemm_scal_openmp
        #define my_dger my_dger_openmp
        #define my_dgemm my_dgemm_openmp
        #define my_dgetf2 my_dgetf2_openmp
        #define my_dgetrf my_dgetrf_openmp
        #define my_dtrsm my_dtrsm_openmp
        #define my_idamax my_idamax_openmp
        #define my_dscal my_dscal_openmp
        #define my_dlaswp my_dlaswp_openmp

        #define my_dgemm_bloc_openmp my_dgemm_openmp // Default version is bloc bersion
    #endif
#endif

} // namespace my_lapack

#ifdef __cplusplus
}
#endif

#endif
