#include <util.h>

#include <stdio.h>
#include <stdlib.h>

void affiche(int m, int n, int* a, int lda){
    if (lda != 0){
        int tmp = n;
        n = m;
        m = tmp;
    }
    int i,j;
    for (i = 0; i < n; i++){
        for (j = 0; j < m; j++){
            printf("\t%d\t", a[i*m +j]); // Pas opti en column major..
        }
        printf("\n");
    }
}

int *mat_alloc(int m, int n){
    return malloc(sizeof(int)*m*n);
};