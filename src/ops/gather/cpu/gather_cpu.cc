#include "gather_cpu.h"
#include "mpi.h"
#include<stdlib.h>

void cpu_gather(void* sendbuff, int send_count, int send_datatype, void* recvbuff, int recv_count, int recv_datatype, int root, Communicator* communicator) {
    MPI_Comm* comm = (MPI_Comm *)communicator;
    MPI_Gather(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, root, *comm);
}