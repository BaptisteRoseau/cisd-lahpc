#pragma once

#include <mpi.h>

class Summa {
  public:
    static Summa &getInstance();
    ~Summa();

    void init( int *argc, char ***argv );
    void reset( int M, int N, int K );
    void finalize();

    int  sizeWorld() const;
    int  rankWorld() const;
    int  rankRow() const;
    int  rankCol() const;
    void gridDimensions( int *r, int *c ) const;
    void A_blockDimensions( int *m, int *n ) const;
    void B_blockDimensions( int *m, int *n ) const;
    void C_blockDimensions( int *m, int *n ) const;

    void sendBlockWorld( int emitter, int receiver, int M, int N, const double *a, int lda, double *b, int ldb ) const;
    int Bcast( double *buffer, int count, int emitter_rank, MPI_Comm communicator ) const;
    MPI_Comm getRowComm() const;
    MPI_Comm getColComm() const;

  private:
    MPI_Comm commRow, commCol;

    int r, c; // grid dimensions

    int *m_a, *n_a;
    int *m_b, *n_b;
    int *m_c, *n_c;

    bool isInitialized;

    int    argc;
    char **argv;

    Summa();
    void freeDimensionsArrays();
};
