#include "send_cpu.h"
#include "mpi.h"
#include<stdlib.h>

void cpu_send(void* sendbuff, int count, int datatype, int peer, Communicator* communicator) {
    MPI_Comm* comm = (MPI_Comm *)communicator;
    MPI_Send(sendbuff, count, datatype, peer, 0, *comm);
}