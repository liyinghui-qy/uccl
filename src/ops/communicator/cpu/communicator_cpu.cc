#include "communicator_cpu.h"
#include "mpi.h"
#include<stdlib.h>

void communicator_cpu_init(){
    MPI_Init(NULL, NULL);
    return;
}

Communicator *get_cpu_communicator(){
    MPI_Comm *comm = (MPI_Comm *)malloc(sizeof(MPI_Comm));
    *comm = MPI_COMM_WORLD;
    return (Communicator *) comm;
}

void get_cpu_commm_size(Communicator* comm, int* size){
    MPI_Comm *mpi_comm = (MPI_Comm *)comm;
    MPI_Comm_size(*mpi_comm, size);
    return;
}

void get_cpu_commm_rank(Communicator* comm, int* rank){
    MPI_Comm *mpi_comm = (MPI_Comm *)comm;
    MPI_Comm_rank(*mpi_comm, rank);
    return;
}
