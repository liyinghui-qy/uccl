#include "broadcast_cpu.h"
#include "mpi.h"
#include<stdlib.h>

void cpu_broadcast(void* buff, int count, int datatype, int root, Communicator* communicator) {
    MPI_Comm* comm = (MPI_Comm*) communicator;
    MPI_Bcast(buff, count, datatype, root, *comm);
}