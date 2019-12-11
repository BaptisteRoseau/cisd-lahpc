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

    void pdgemm( int           m,
                 int           n,
                 int           k,
                 int           nb,
                 double        alpha,
                 const double *a,
                 int           lda,
                 const double *b,
                 int           ldb,
                 double        beta,
                 double *      c,
                 int           ldc,
                 int *         m_a,
                 int *         n_a,
                 int *         m_b,
                 int *         n_b,
                 int *         m_c,
                 int *         n_c,
                 MPI_Comm      comm_row,
                 MPI_Comm      comm_col,
                 double *      work1,
                 double *      work2 )
    {
        int     myrow, mycol, nprow, npcol, i, j, kk, iwrk, icurrow, icurcol, ii, jj;
        double *temp;
        double *p;

        Summa &SUMMA = Summa::getInstance();

        MPI_Comm_rank( comm_row, &mycol );
        MPI_Comm_rank( comm_col, &myrow );

        for ( j = 0; j < n_c[mycol]; j++ ) {
            for ( i = 0; i < m_c[myrow]; i++ ) {
                C( i, j ) = beta * C( i, j );
            }
        }
        icurrow = 0;
        icurcol = 0;
        ii = jj = 0;
        // temp    = new double[ m_c[myrow] * nb ]; useless ? WTF ???
        for ( kk = 0; kk < k; kk += iwrk ) {
            iwrk = std::min( nb, m_b[icurrow] - ii );
            iwrk = std::min( iwrk, n_a[icurcol] - jj );

            if ( mycol == icurcol ) my_dlacpy( m_a[myrow], iwrk, &A( 0, jj ), lda, work1, m_a[myrow] );
            if ( myrow == icurrow ) my_dlacpy( iwrk, n_b[mycol], &B( ii, 0 ), ldb, work2, iwrk );

            /* broadcast work1 and work2*/
            SUMMA.Bcast( work1, m_a[myrow] * iwrk, icurcol, comm_row );
            SUMMA.Bcast( work2, n_b[mycol] * iwrk, icurrow, comm_col );

            my_dgemm_seq( CblasColMajor,
                          CblasNoTrans,
                          CblasNoTrans,
                          m_c[myrow],
                          n_c[mycol],
                          iwrk,
                          alpha,
                          work1,
                          m_b[myrow],
                          work2,
                          iwrk,
                          1.,
                          c,
                          ldc );

            ii += iwrk;
            jj += iwrk;
            if ( jj >= n_a[icurcol] ) {
                icurcol++;
                jj = 0;
            };
            if ( ii >= m_b[icurrow] ) {
                icurrow++;
                ii = 0;
            };
        }
        // delete[] temp; ???
    }

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

        // Create blocks
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

            for ( int i = 0; i < proc % colCount;
                  ++i ) { // TODO : "i < proc % colCount" that's a clear fuck to encapsulation
                A_assignedWidth += n_a[i];
                B_assignedWidth += n_b[i];
                C_assignedWidth += n_c[i];
            }

            for ( int i = 0; i < proc / rowCount;
                  ++i ) { // TODO : "i < proc / colCount" that's a clear fuck to encapsulation
                A_assignedHeight += m_a[i];
                B_assignedHeight += m_b[i];
                C_assignedHeight += m_c[i];
            }
            if ( rankWorld == 0 )
                std::cout << "A( " << A_assignedHeight << ", " << A_assignedWidth << " )" << std::endl;
            SUMMA.sendBlockWorld( 0,
                                  proc,
                                  m_a[rankRow],
                                  n_a[rankCol],
                                  &A( A_assignedHeight, A_assignedWidth ),
                                  lda,
                                  A_block.data(),
                                  m_a[rankRow] );
            SUMMA.sendBlockWorld( 0,
                                  proc,
                                  m_b[rankRow],
                                  n_b[rankCol],
                                  &B( B_assignedHeight, B_assignedWidth ),
                                  ldb,
                                  B_block.data(),
                                  m_b[rankRow] );
            SUMMA.sendBlockWorld( 0,
                                  proc,
                                  m_c[rankRow],
                                  n_c[rankCol],
                                  &C( C_assignedHeight, C_assignedWidth ),
                                  ldc,
                                  C_block.data(),
                                  m_c[rankRow] );
        }

        /*if ( rankWorld == 3 ) {
            std::cout << "rank col: " << rankCol << " rank row: " << rankRow << std::endl;
            affiche( m_a[rankRow], n_a[rankCol], A_block.data(), m_a[rankRow], std::cout );
        }*/

        // Create work arrays
        int     nb    = n_a[rankCol]; // TODO : is this correct ?
        double *work1 = new double[nb * m_a[rankRow]];
        double *work2 = new double[nb * n_b[rankCol]];

        // call pdgemm
        pdgemm( M,
                N,
                K,
                nb,
                alpha,
                A_block.data(),
                m_a[rankRow],
                B_block.data(),
                m_b[rankRow],
                beta,
                C_block.data(),
                m_c[rankRow],
                m_a.data(),
                n_a.data(),
                m_b.data(),
                n_b.data(),
                m_c.data(),
                n_c.data(),
                SUMMA.getRowComm(),
                SUMMA.getColComm(),
                work1,
                work2 );

        // Gather blocks
        if ( rankWorld == 2 ) {
            std::cout << "rank col: " << rankCol << " rank row: " << rankRow << std::endl;
            affiche( m_c[rankRow], n_c[rankCol], C_block.data(), m_c[rankRow], std::cout );
        }

        /*for ( int proc = 0; proc < worldSize; ++proc ) {
            int C_assignedWidth = 0, C_assignedHeight = 0;

            for ( int i = 0; i < proc % colCount; ++i ) {
                C_assignedWidth += n_a[i];
            }

            for ( int i = 0; i < proc / rowCount; ++i ) {
                C_assignedHeight += m_a[i];
            }

            SUMMA.sendBlockWorld( proc,
                                  0,
                                  m_c[rankRow],
                                  n_c[rankCol],
                                  C_block.data(),
                                  m_c[rankRow],
                                  &C( C_assignedHeight, C_assignedWidth ),
                                  ldc );
        }*/

        if ( rankWorld == 0 ) {
            std::cout << "rank col: " << rankCol << " rank row: " << rankRow << std::endl;
            affiche( M, N, c, ldc, std::cout );
        }

        // Leave
        delete[] work1, work2;
    }
} // namespace my_lapack
