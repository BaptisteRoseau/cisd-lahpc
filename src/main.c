#include <my_lapack.h>
#include <util.h>

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
//#include <flops.h>
//#include <lapacke.h> //header include error
//#include <perf.h>

#define MAT_WIDTH 5
#define MAT_HEIGHT 7


int main(int argc, char **argv){
    int *mat = malloc(sizeof(int)*MAT_WIDTH*MAT_HEIGHT);

    /* Matrix Initialization */
    int i,j; 
    for (i = 0; i < MAT_HEIGHT; i++){
        for (j = 0; j < MAT_WIDTH; j++){
            mat[i*MAT_WIDTH + j] = i*MAT_WIDTH + j;
        }   
    }

    /* Basic functions call */
    affiche(MAT_WIDTH, MAT_HEIGHT, mat, 0);

    free(mat);
    return 0;
}