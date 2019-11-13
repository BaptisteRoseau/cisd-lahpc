#pragma once

namespace my_lapack {

    class Mat {
      public:
        Mat() = delete;
        ~Mat();
        Mat( int m, int n );
        Mat( int m, int n, double value );
        Mat( const Mat &other );

        Mat &operator=( const Mat &other );
        bool operator==( const Mat &other );

        inline double &at( int i, int j ) const { return storage[j * m + i]; };
        inline double *get() { return storage; }
        double* col(int j);
        inline int     dimX() { return m; }
        inline int     dimY() { return n; }

      private:
        double *storage;
        int     m, n;

        double *initStorage( int size );
    };

    Mat MatRandi( int m, int n, unsigned int max, unsigned int seed = 0x9d2c5680 );
    Mat MatSqrDiag( int m, double v );

} // namespace my_lapack
