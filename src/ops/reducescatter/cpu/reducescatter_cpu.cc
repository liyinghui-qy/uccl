#include "reducescatter_cpu.h"
#include "mpi.h"
#include<stdlib.h>

void cpu_reducescatter(void* sendbuff, void* recvbuff, int* recvcounts, int datatype, int op, Communicator* communicator) {
    MPI_Comm* comm = (MPI_Comm *)communicator;
    MPI_Reduce_scatter(sendbuff, recvbuff, recvcounts, datatype, op, *comm);
}