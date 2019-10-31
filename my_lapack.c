#include <stdio.h>
#include <cblas.h>
//#include <flops.h>
//#include <lapacke.h> //header include error
//#include <perf.h>

#define MAT_WIDTH 5
#define MAT_HEIGHT 7

enum lda{
    RAW_MAJOR, COL_MAJOR
};

double my_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);


void my_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);

int my_dgetrf(const enum CBLAS_ORDER Order, int m, int n, double* a, int lda, int* ipiv);


void affiche(int m, int n, int* a, const enum lda lda){
    if (lda == COL_MAJOR){
        int tmp = n;
        n = m;
        m = tmp;
    }
    int i,j;
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            printf(" %d ", a[i*n +j]);
        }
        printf("\n");
    }
}



int main(int argc, char **argv){
    int *mat = malloc(sizeof(int)*MAT_WIDTH*MAT_HEIGHT);


    free(mat);
    return 0;
}