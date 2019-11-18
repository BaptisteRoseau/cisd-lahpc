#ifndef UTIL_H
#define UTIL_H

#include <ostream>

namespace my_lapack {

    /* lda = 0 : column major (else: row major) */
    void affiche( int m, int n, const double *a, int lda, std::ostream &stream, int precision = 6);

} // namespace my_lapack

#endif
