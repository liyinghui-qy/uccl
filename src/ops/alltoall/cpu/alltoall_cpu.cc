#include "alltoall_cpu.h"
#include "mpi.h"
#include<stdlib.h>
#include "../../../devices/cpu/common_cpu.h"

void cpu_alltoall(void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, Communicator* communicator) {
    MPI_Comm* comm = (MPI_Comm *)communicator;
    MPI_Datatype send_datatype_mpi = ccl_to_mpi_datatype(send_datatype);
    MPI_Datatype recv_datatype_mpi = ccl_to_mpi_datatype(recv_datatype);
    MPI_Alltoall(sendbuff, send_count, send_datatype_mpi, recvbuff, recv_count, recv_datatype_mpi, *comm);
}