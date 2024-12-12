#include "alltoall_cpu.h"
#include "mpi.h"
#include<stdlib.h>

void cpu_alltoall(void* sendbuff, int send_count, int send_datatype, void* recvbuff, int recv_count, int recv_datatype, Communicator* communicator) {
    MPI_Comm* comm = (MPI_Comm *)communicator;
    MPI_Alltoall(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, *comm);
}