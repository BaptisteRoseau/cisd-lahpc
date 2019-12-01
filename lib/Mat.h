#pragma once

#include <limits>

namespace my_lapack {

    class Mat {
      public:
        Mat();
        ~Mat();
        Mat( int m, int n );
        Mat( int m, int n, double value );
        Mat( const Mat &other );
        void reshape( int m, int n, double value );

        Mat &operator=( const Mat &other );
        bool operator==( const Mat &other );

        inline double &at( int i, int j ) const { return storage[j * m + i]; };
        inline double &at( int i ) const { return storage[i]; };
        inline double *get() { return storage; }
        double *       col( int j );
        inline int     dimX() { return m; }
        inline int     dimY() { return n; }
        inline int     numRow() { return m; }
        inline int     numCol() { return n; }
        inline int     ld() { return m; }
        void           print( int precision = 6 );
        bool           equals( const Mat &m, double epsilon = std::numeric_limits<double>::epsilon() );
        bool           containsOnly( const double d );
        void           fill( double d );

      private:
        double *storage;
        int     m, n;

        double *initStorage( int size );
    };

    Mat MatRandi( int m, int n, unsigned int max, unsigned int seed = 0x9d2c5680 );
    Mat MatSqrDiag( int m, double v );
    Mat MatZero( int m, int n );
    Mat MatRandLi( int m );
    Mat MatRandUi( int m );

} // namespace my_lapack
