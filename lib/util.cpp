#include "util.h"

#include <iomanip>

namespace my_lapack {

    void affiche( int m, int n, const double *a, int lda, std::ostream &stream, int precision )
    {
        auto oldPrecision = stream.precision( precision );

        if ( lda ) {
            for ( int j = 0; j < n; ++j ) {
                stream << "| ";
                for ( int i = 0; i < m; ++i ) { stream << a[i * n + j] << " "; }
                stream << "|\n";
            }
        }

        else {
            for ( int i = 0; i < m; ++i ) {
                stream << "| ";
                for ( int j = 0; j < n; ++j ) { stream << a[j * m + i] << " "; }
                stream << "|\n";
            }
        }

        stream << std::endl;
        stream.precision( oldPrecision );
    }

    int dequals(const double a, const double b){
        return std::abs(a-b) < LAHPC_EPSILON;
    }

} // namespace my_lapack