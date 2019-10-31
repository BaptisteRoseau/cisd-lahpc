#include <my_lapack.h>

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


    free(mat);
    return 0;
}