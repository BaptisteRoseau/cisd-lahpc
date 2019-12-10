#include "Summa.hpp"

#include "err.h"
#include "my_lapack.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#define A( i, j ) ( a[j * lda + i] )
#define B( i, j ) ( b[j * ldb + i] )
#define C( i, j ) ( c[j * ldc + i] )

Summa::Summa()
    : commRow( 0 )
    , commCol( 0 )
    , r( -1 )
    , c( -1 )
    , m_a( nullptr )
    , n_a( nullptr )
    , m_b( nullptr )
    , n_b( nullptr )
    , m_c( nullptr )
    , n_c( nullptr )
    , isInitialized( false )
    , argc( 0 )
    , argv( nullptr )
{
}

void throwNotInitialized()
{
    throw std::exception( "Summa is not initialized" );
}

void Summa::init( int *argc, char ***argv )
{
    LAHPC_CHECK_PREDICATE( argc != nullptr && argv != nullptr );

    this->argc = *argc;
    this->argv = new char *[*argc];
    for ( int i = 0; i < *argc; ++i ) {
        auto len      = std::strlen( *argv[i] );
        this->argv[i] = new char[len];
        std::strcpy( this->argv[i], *argv[i] );
    }

    isInitialized = true;
}

// tODO : no need to do this if M and N have not changed since last call
void Summa::reset( int M, int N, int K )
{
    if ( isInitialized ) {
        MPI_Init( &argc, &argv );

        int worldSize, worldRank;
        MPI_Comm_size( MPI_COMM_WORLD, &worldSize );
        MPI_Comm_rank( MPI_COMM_WORLD, &worldRank );

        // Make sure the number of processes is the square of some number
        int squareSide = std::sqrt( worldSize );
        if ( squareSide * squareSide != worldSize ) {
            throw std::invalid_argument( "The number of processes must be the square of some positive number" );
        }

        this->r = this->c = squareSide;

        freeDimensionsArrays();
        m_a = new int[r];
        m_b = new int[r];
        m_c = new int[r];
        n_a = new int[c];
        n_b = new int[c];
        n_c = new int[c];

        int A_blockRowCount = M / r, A_blockColCount = K / c;
        int B_blockRowCount = K / r, B_blockColCount = N / c;
        int C_blockRowCount = M / r, C_blockColCount = N / c;
        
        for ( int i = 0; i < r; ++i ) {
            m_a[i] = A_blockRowCount;
            m_b[i] = B_blockRowCount;
            m_c[i] = C_blockRowCount;
        }
        // leftovers
        m_a[r - 1] += M % r;
        m_b[r - 1] += K % r;
        m_c[r - 1] += M % r;

        for (int i = 0; i < c; ++i)
        {
            n_a[i] = A_blockColCount;
            n_b[i] = B_blockColCount;
            n_c[i] = C_blockColCount;
        }
        // leftovers
        n_a[c - 1] += K % r;
        n_b[c - 1] += N % c;
        n_c[c - 1] += N % c;

        // row communicator
        MPI_Comm_split( MPI_COMM_WORLD, worldRank / squareSide, worldRank, &commRow );
        // column communicator
        MPI_Comm_split( MPI_COMM_WORLD, worldRank % squareSide, worldRank, &commCol );
    }
    else {
        throwNotInitialized();
    }
}

void Summa::finalize()
{
    MPI_Finalize();
}

template<typename T>
void deleteArray( T **ptr )
{
    if ( *ptr != nullptr ) {
        delete[] * ptr;
        *ptr = nullptr;
    }
}

Summa::~Summa()
{
    MPI_Finalize();
    for ( int i = 0; i < argc; ++i ) {
        deleteArray( &argv[i] );
    }
    deleteArray( &argv );

    freeDimensionsArrays();
}

int Summa::rankWorld() const
{
    if ( isInitialized ) {
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        return rank;
    }
    else {
        throwNotInitialized();
    }
}

int Summa::rankRow() const
{
    if ( isInitialized ) {
        int rank;
        MPI_Comm_rank( commRow, &rank );
        return rank;
    }
    else {
        throwNotInitialized();
    }
}

int Summa::rankCol() const
{
    if ( isInitialized ) {
        int rank;
        MPI_Comm_rank( commCol, &rank );
        return rank;
    }
    else {
        throwNotInitialized();
    }
}

void Summa::sendBlock( MPI_Comm      communicator,
                       int           emitter,
                       int           receiver,
                       int           M,
                       int           N,
                       const double *a,
                       int           lda,
                       double *      b,
                       int           ldb )
{
    int rankWorld_ = rankWorld();
    if ( rankWorld_ == emitter ) {
        double *sendBlock = new double[M * N];
        my_lapack::my_dlacpy( M, N, a, lda, sendBlock, ldb );
        MPI_Send( sendBlock, M * N, MPI_DOUBLE, receiver, 0, communicator );
    }
    else if ( rankWorld_ == receiver ) {
        MPI_Status status;
        MPI_Recv( b, M * N, MPI_DOUBLE, emitter, 0, communicator, &status );
    }
}

int Summa::Bcast( double *buffer, int count, int emitter_rank, MPI_Comm communicator )
{
    if ( isInitialized ) { MPI_Bcast( static_cast<void *>( buffer ), count, MPI_DOUBLE, emitter_rank, communicator ); }
    else {
        throwNotInitialized();
    }
}

void Summa::freeDimensionsArrays()
{
    deleteArray( &m_a );
    deleteArray( &m_b );
    deleteArray( &m_c );
    deleteArray( &n_a );
    deleteArray( &n_b );
    deleteArray( &n_c );
}
