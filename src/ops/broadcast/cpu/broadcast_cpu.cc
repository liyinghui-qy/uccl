#include "broadcast_cpu.h"
#include "mpi.h"
#include<stdlib.h>
#include "../../../devices/cpu/common_cpu.h"

void cpu_broadcast(void* buff, int count, CCLDatatype datatype, int root, Communicator* communicator) {
    MPI_Comm* comm = (MPI_Comm*) communicator;
    MPI_Datatype datatype_mpi = ccl_to_mpi_datatype(datatype);
    MPI_Bcast(buff, count, datatype_mpi, root, *comm);
}