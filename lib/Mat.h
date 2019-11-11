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

        inline double &at( int i, int j ) const { return storage[j * m + i]; };
        inline double *get() { return storage; }
        inline int dimX() { return m; }
        inline int dimY() { return n; }

      private:
        double *storage;
        int     m, n;

        double *initStorage( int size );
    };

} // namespace my_lapack
