#include "reducescatter_cpu.h"
#include "mpi.h"
#include<stdlib.h>
#include "../../../devices/cpu/common_cpu.h"

void cpu_reducescatter(void* sendbuff, void* recvbuff, int* recvcounts, CCLDatatype datatype, CCLOp op, Communicator* communicator) {
    MPI_Comm* comm = (MPI_Comm *)communicator;
    MPI_Datatype datatype_mpi = ccl_to_mpi_datatype(datatype);
    MPI_Op op_mpi = ccl_to_mpi_op(op);
    MPI_Reduce_scatter(sendbuff, recvbuff, recvcounts, datatype_mpi, op_mpi, *comm);
}