#ifndef __CPU_ALLGATHER_H__
#define __CPU_ALLGATHER_H__

#include "operators.h"
#include "ops/allgather/allgather.h"

typedef struct AllgatherCpuDescriptor {
    Device device;
} AllgatherCpuDescriptor;

void cpu_allgather(void* sendbuff, int send_count, int send_datatype, void* recvbuff, int recv_count, int recv_datatype, Communicator* communicator);

#endif