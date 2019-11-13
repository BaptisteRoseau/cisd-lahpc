#pragma once

#include <stdexcept>

#ifdef LAHPC_DEBUG
    #define LAHPC_CHECK_POSITIVE( X ) \
        if ( X < 0 ) { throw std::domain_error( #X " should be positive." ); }

    #define LAHPC_CHECK_POSITIVE_STRICT( X ) \
        if ( X <= 0 ) { throw std::domain_error( #X " should be strictly positive." ); }

#else
    #define LAHPC_CHECK_POSITIVE( X )
    #define LAHPC_CHECK_POSITIVE_STRICT( X )
#endif

#define LAHPC_CHECK_PREDICATE( predicate ) \
    if ( !( predicate ) ) { throw std::logic_error( "Predicate " #predicate " evaluates to false." ); }
