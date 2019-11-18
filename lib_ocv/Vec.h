#pragma once

#include <cstdint>

class Vec;

class Vec {
  public:
    Vec();
    Vec( uint32_t dim);
    Vec( uint32_t dim, double value );
    Vec( const Vec &other );

    Vec &operator=( const Vec &other );
    Vec &operator[]( uint32_t i );

    inline double at( uint32_t i, uint32_t j ) const;

  private:
    double *    storage;
    uint32_t dim;

};