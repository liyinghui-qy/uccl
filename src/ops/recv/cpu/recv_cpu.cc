#include "recv_cpu.h"
#include "mpi.h"
#include<stdlib.h>

void cpu_recv(void* recvbuff, int count, int datatype, int peer, Communicator* communicator, Status* status) {
    MPI_Comm* comm = (MPI_Comm *)communicator;
    MPI_Recv(recvbuff, count, (MPI_Datatype)datatype, peer, 0, *comm, MPI_STATUS_IGNORE);
}