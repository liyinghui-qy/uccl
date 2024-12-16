#include "send_cpu.h"
#include "mpi.h"
#include<stdlib.h>
#include "../../../devices/cpu/common_cpu.h"

void cpu_send(void* sendbuff, int count, CCLDatatype datatype, int peer, Communicator* communicator) {
    MPI_Comm* comm = (MPI_Comm *)communicator;
    MPI_Datatype dt = ccl_to_mpi_datatype(datatype);
    MPI_Send(sendbuff, count, dt, peer, 0, *comm);
}