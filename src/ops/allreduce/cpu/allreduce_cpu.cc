#include "allreduce_cpu.h"
#include "mpi.h"
#include<stdlib.h>

void cpu_allreduce(void* sendbuff, void* recvbuff, int count, int datatype, int op, Communicator* communicator) {
    MPI_Comm* comm = (MPI_Comm*) communicator;
    MPI_Allreduce(sendbuff, recvbuff, count, datatype, op, *comm);
}