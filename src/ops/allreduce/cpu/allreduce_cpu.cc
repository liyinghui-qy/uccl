#include "allreduce_cpu.h"
#include "mpi.h"
#include<stdlib.h>
#include "../../../devices/cpu/common_cpu.h"

void cpu_allreduce(void* sendbuff, void* recvbuff, int count, CCLDatatype datatype, CCLOp op, Communicator* communicator) {
    MPI_Comm* comm = (MPI_Comm*) communicator;
    MPI_Datatype datatype_mpi = ccl_to_mpi_datatype(datatype);
    MPI_Op op_mpi = ccl_to_mpi_op(op);
    MPI_Allreduce(sendbuff, recvbuff, count, datatype_mpi, op_mpi, *comm);
}