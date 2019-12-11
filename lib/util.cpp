#include "util.h"

#include <cmath>
#include <iomanip>

namespace my_lapack {

    void affiche( int m, int n, const double *a, int lda, std::ostream &stream, int precision )
    {
        auto oldPrecision = stream.precision( precision );
        for ( int j = 0; j < n; ++j ) {
            stream << "| ";
            for ( int i = 0; i < m; ++i ) {
                stream << a[j * lda + i] << " ";
            }
            stream << "|\n";
        }

        stream << std::endl;
        stream.precision( oldPrecision );
    }

    int dequals( const double a, const double b, const double epsilon ) { return std::abs( a - b ) < epsilon; }

} // namespace my_lapack