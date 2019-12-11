#include "Summa.hpp"
#include "err.h"
#include "my_lapack.h"
#include "util.h"

#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <vector>

/* Access matrix elements provided storage is COLUMN MAJOR */
#define A( i, j ) ( a[j * lda + i] )
#define B( i, j ) ( b[j * ldb + i] )
#define C( i, j ) ( c[j * ldc + i] )

namespace my_lapack {

    void my_dlacpy( int M, int N, const double *a, int lda, double *b, int ldb )
    {
        LAHPC_CHECK_POSITIVE( M );
        LAHPC_CHECK_POSITIVE( N );
        LAHPC_CHECK_PREDICATE( lda >= std::max( 1, M ) );
        LAHPC_CHECK_PREDICATE( ldb >= std::max( 1, M ) );

        for ( int j = 0; j < N; ++j ) {
            for ( int i = 0; i < M; ++i ) {
                B( i, j ) = A( i, j );
            }
        }
    }

    void init_lapack_mpi( int *argc, char ***argv ) { Summa::getInstance().init( argc, argv ); }

    void my_dgemm_mpi( CBLAS_ORDER     Order,
                       CBLAS_TRANSPOSE TransA,
                       CBLAS_TRANSPOSE TransB,
                       int             M,
                       int             N,
                       int             K,
                       double          alpha,
                       const double *  a,
                       int             lda,
                       const double *  b,
                       int             ldb,
                       double          beta,
                       double *        c,
                       int             ldc )
    {
        // SUMMA.reset( M, N, K ); Must have been done before !

        Summa &SUMMA = Summa::getInstance();
        // SUMMA.reset( M, N, K );

        int worldSize = SUMMA.sizeWorld();
        int rankWorld = SUMMA.rankWorld();
        int rankRow   = SUMMA.rankRow();
        int rankCol   = SUMMA.rankCol();
        int rowCount, colCount;
        SUMMA.gridDimensions( &rowCount, &colCount );

        // std::cout << "World size : " << worldSize << std::endl;

        std::vector<int> m_a( rowCount ), n_a( colCount ), m_b( rowCount ), n_b( colCount ), m_c( rowCount ),
            n_c( colCount );

        // Creating blocks
        SUMMA.A_blockDimensions( m_a.data(), n_a.data() );
        SUMMA.B_blockDimensions( m_b.data(), n_b.data() );
        SUMMA.C_blockDimensions( m_c.data(), n_c.data() );

        std::vector<double> A_block( m_a[rankRow] * n_a[rankCol] );
        std::vector<double> B_block( m_b[rankRow] * n_b[rankCol] );
        std::vector<double> C_block( m_c[rankRow] * n_c[rankCol] );

        for ( int proc = 0; proc < worldSize; ++proc ) {
            int A_assignedWidth = 0, A_assignedHeight = 0;
            int B_assignedWidth = 0, B_assignedHeight = 0;
            int C_assignedWidth = 0, C_assignedHeight = 0;

            for ( int i = 0; i < rankRow; ++i ) {
                A_assignedWidth += n_a[i];
                B_assignedWidth += n_b[i];
                C_assignedWidth += n_c[i];
            }

            for ( int i = 0; i < rankCol; ++i ) {
                A_assignedHeight += m_a[i];
                B_assignedHeight += m_b[i];
                C_assignedHeight += m_c[i];
            }

            SUMMA.sendBlockWorld( 0,
                                  proc,
                                  m_a[rankRow],
                                  n_a[rankCol],
                                  &A( A_assignedWidth, A_assignedHeight ),
                                  lda,
                                  A_block.data(),
                                  m_a[rankRow] );
            SUMMA.sendBlockWorld( 0,
                                  proc,
                                  m_b[rankRow],
                                  n_b[rankCol],
                                  &B( B_assignedWidth, B_assignedHeight ),
                                  ldb,
                                  B_block.data(),
                                  m_b[rankRow] );
            SUMMA.sendBlockWorld( 0,
                                  proc,
                                  m_c[rankRow],
                                  n_c[rankCol],
                                  &C( C_assignedWidth, C_assignedHeight ),
                                  ldc,
                                  C_block.data(),
                                  m_c[rankRow] );
        }

        if ( rankWorld == 0 ) { affiche( m_a[rankRow], n_a[rankCol], A_block.data(), m_a[rankRow], std::cout ); }

        // SUMMA.finalize(); Must be done at some point !
    }
} // namespace my_lapack
