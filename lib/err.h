#pragma once

#include <stdexcept>

#if defined( LAHPC_DEBUG ) && !defined( LAHPC_NO_CHECK )
    #define LAHPC_CHECK_POSITIVE( X ) \
        if ( X < 0 ) { throw std::domain_error( #X " should be positive." ); }

    #define LAHPC_CHECK_POSITIVE_STRICT( X ) \
        if ( X <= 0 ) { throw std::domain_error( #X " should be strictly positive." ); }

    #define LAHPC_CHECK_PREDICATE( predicate ) \
        if ( !( predicate ) ) { throw std::logic_error( "Predicate " #predicate " evaluates to false." ); }

#else

    #define LAHPC_CHECK_POSITIVE( X )
    #define LAHPC_CHECK_POSITIVE_STRICT( X )
    #define LAHPC_CHECK_PREDICATE ( predicate )

#endif
