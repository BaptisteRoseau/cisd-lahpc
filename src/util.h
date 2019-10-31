#ifndef UTIL_H
#define UTIL_H

/*===================== Custom functions for testing purpose */

/* lda = 0 : column major (else: raw major) */
void affiche(int m, int n, int* a, int lda);

int *mat_alloc(int m, int n);

#endif
