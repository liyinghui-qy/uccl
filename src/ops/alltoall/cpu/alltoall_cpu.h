#ifndef __CPU_ALLTOALL_H__
#define __CPU_ALLTOALL_H__

#include "operators.h"
#include "ops/alltoall/alltoall.h"

typedef struct AlltoallCpuDescriptor {
    Device device;
} AlltoallCpuDescriptor;

void cpu_alltoall(void* sendbuff, int send_count, int send_datatype, void* recvbuff, int recv_count, int recv_datatype, Communicator* communicator);

#endif