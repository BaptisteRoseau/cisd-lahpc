#include "algonum.h"

void
 my_dgemm_seq( CBLAS_LAYOUT layout,
	       CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
	       int M, int N, int K,
	       double alpha, const double *A, int lda,
                             const double *B, int ldb,
	       double beta,        double *C, int ldc )
{

}
