#pragma once

#include <mpi.h>

class Summa {
  public:
    Summa();
    void init( int *argc, char ***argv );
    void reset( int M, int N, int K );
    void finalize();
    ~Summa();

    int rankWorld() const;
    int rankRow() const;
    int rankCol() const;

    void sendBlock( MPI_Comm      communicator,
                    int           emitter,
                    int           receiver,
                    int           M,
                    int           N,
                    const double *a,
                    int           lda,
                    double *      b,
                    int           ldb );
    int  Bcast( double *buffer, int count, int emitter_rank, MPI_Comm communicator );
    void A_blockDimensions( int **m, int **n )
    {
        *m = m_a;
        *n = n_a;
    }
    void B_blockDimensions( int **m, int **n )
    {
        *m = m_b;
        *n = n_b;
    }
    void C_blockDimensions( int **m, int **n )
    {
        *m = m_c;
        *n = n_c;
    }

  private:
    MPI_Comm commRow, commCol;

    int r, c; // grid dimensions

    int *m_a, *n_a;
    int *m_b, *n_b;
    int *m_c, *n_c;

    bool isInitialized;

    int    argc;
    char **argv;

    void freeDimensionsArrays();
};
