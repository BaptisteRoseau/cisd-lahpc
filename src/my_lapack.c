#include <stdio.h>
#include <cblas.h>
//#include <flops.h>
//#include <lapacke.h> //header include error
//#include <perf.h>

void affiche(int m, int n, int* a, int lda){
    if (lda == 0){
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
