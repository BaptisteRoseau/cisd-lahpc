#include "Mat.h"

#include <cstring>
#include <iostream>
#include <new>
#include <random>

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
        if ( value == 0.0 ) {
            memset( storage, 0, static_cast<std::size_t>( m ) * static_cast<std::size_t>( n ) * sizeof( double ) );
        }
        else {
            for ( int i = 0; i < m * n; ++i ) { storage[i] = value; }
        }
    }

    Mat::Mat( const Mat &other )
        : m( other.m )
        , n( other.n )
    {
        storage = initStorage( m * n );
        memcpy( storage,
                other.storage,
                static_cast<std::size_t>( m ) * static_cast<std::size_t>( n ) * sizeof( double ) );
    }

    double *Mat::col( int j ) { return storage + static_cast<std::size_t>( j ) * static_cast<std::size_t>( m ); }

    Mat &Mat::operator=( const Mat &other )
    {
        if ( &other != this ) {
            // I chose to offer the strong exception safety.
            // Though it eats up memory...
            double *tmp = initStorage( other.m * other.n );
            for ( int i = 0; i < other.m * other.n; ++i ) { tmp[i] = other.storage[i]; }

            delete[] storage;
            storage = tmp;
            m       = other.m;
            n       = other.n;
        }

        return *this;
    }

    bool Mat::operator==( const Mat &other )
    {
        return m == other.m && n == other.n &&
               ( memcmp( storage,
                         other.storage,
                         static_cast<std::size_t>( m ) * static_cast<std::size_t>( n ) * sizeof( double ) ) == 0 );
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

    Mat MatRandi( int m, int n, unsigned int max, unsigned int seed /*= 0x9d2c5680*/ )
    {
        static std::minstd_rand randEngine;

        randEngine.seed( seed );

        Mat mat( m, n );
        for ( int j = 0; j < n; ++j ) {
            for ( int i = 0; i < m; ++i ) { mat.at( i, j ) = randEngine() % ( max + 1 ); }
        }

        return mat;
    }

    Mat MatSqrDiag( int m, double v )
    {
        Mat mat( m, m, 0 );
        for ( int i = 0; i < m; ++i ) { mat.at( i, i ) = v; }

        return mat;
    }

} // namespace my_lapack