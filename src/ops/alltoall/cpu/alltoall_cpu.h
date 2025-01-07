#ifndef __CPU_ALLTOALL_H__
#define __CPU_ALLTOALL_H__

#include "ops/alltoall/alltoall.h"

typedef struct AlltoallCpuDescriptor {
    Device device;
} AlltoallCpuDescriptor;

void cpu_alltoall(void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, Communicator* communicator);

#endif