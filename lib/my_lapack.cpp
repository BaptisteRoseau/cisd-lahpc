#include "my_lapack.h"

#include <cstdint>

#ifdef LAHPC_DEBUG
    #include <stdexcept>

    #define LAHPC_CHECK_POSITIVE( X ) \
        if ( X < 0 ) { throw std::domain_error( #X " should be positive" ); }
#endif

namespace my_lapack {

    double my_ddot( const int N, const double *X, const int incX, const double *Y, const int incY )
    {
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_POSITIVE( incX );
        LAHPC_CHECK_POSITIVE( incY );

        double ret = 0;
        for ( uint32_t i = 0, xi = 0, yi = 0; i < N; ++i, xi += incX, yi += incY ) { ret += X[xi] * Y[yi]; }
        return ret;
    }

} // namespace my_lapack