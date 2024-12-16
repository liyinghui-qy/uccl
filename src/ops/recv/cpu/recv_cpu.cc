#include "recv_cpu.h"
#include "mpi.h"
#include<stdlib.h>
#include "../../../devices/cpu/common_cpu.h"

void cpu_recv(void* recvbuff, int count, CCLDatatype datatype, int peer, Communicator* communicator, Status* status) {
    MPI_Comm* comm = (MPI_Comm *)communicator;
    MPI_Datatype dt = ccl_to_mpi_datatype(datatype);
    MPI_Recv(recvbuff, count, dt, peer, 0, *comm, MPI_STATUS_IGNORE);
}