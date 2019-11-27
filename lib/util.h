#ifndef UTIL_H
#define UTIL_H

#include <ostream>

#define LAHPC_EPSILON 1e-8

namespace my_lapack {

    /* lda = 0 : column major (else: row major) */
    void affiche( int m, int n, const double *a, int lda, std::ostream &stream, int precision = 6 );

    int dequals( const double a, const double b, const double epsilon );

} // namespace my_lapack

#endif
