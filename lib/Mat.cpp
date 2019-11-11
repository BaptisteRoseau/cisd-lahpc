#include "Mat.h"

#include <cstring>
#include <iostream>
#include <new>

namespace my_lapack {

    Mat::~Mat() { delete[] storage; }

    Mat::Mat( int m, int n )
        : m( m )
        , n( n )
    {
        storage = initStorage( m * n );
    }

    Mat::Mat( int m, int n, double value )
        : m( m )
        , n( n )
    {
        storage = initStorage( m * n );
        if ( value == 0.0 ) { memset( storage, 0, m * n * sizeof( double ) ); }
        else {
            for ( int i = 0; i < m * n; ++i ) { storage[i] = value; }
        }
    }

    Mat::Mat( const Mat &other )
        : m( other.m )
        , n( other.n )
    {
        storage = initStorage( m * n );
        memcpy( storage, other.storage, m * n * sizeof( double ) );
    }

    Mat &Mat::operator=( const Mat &other )
    {
        // I chose to offer the strong exception safety.
        // Though it eats up memory...
        double *tmp = initStorage( other.m * other.n );
        for ( int i = 0; i < other.m * other.n; ++i ) { tmp[i] = other.storage[i]; }

        delete[] storage;
        storage = tmp;
        m       = other.m;
        n       = other.n;

        return *this;
    }

    double *Mat::initStorage( int size )
    {
        try {
            return new double[size];
        }
        catch ( const std::bad_alloc &e ) {
            std::cerr << "ERROR::Mat::Mat()\n" << e.what() << std::endl;
            throw e;
        }
    }

} // namespace my_lapack