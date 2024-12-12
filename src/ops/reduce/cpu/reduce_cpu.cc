#include "reduce_cpu.h"
#include "mpi.h"
#include<stdlib.h>

void cpu_reduce(void* sendbuff, void* recvbuff, int count, int datatype, int op, int root, Communicator* communicator) {
    MPI_Comm* comm = (MPI_Comm *)communicator;
    MPI_Reduce(sendbuff, recvbuff, count, datatype, op, root, *comm);
}