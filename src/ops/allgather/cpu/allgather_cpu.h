#ifndef __CPU_ALLGATHER_H__
#define __CPU_ALLGATHER_H__

#include "ops/allgather/allgather.h"

typedef struct AllgatherCpuDescriptor {
    Device device;
} AllgatherCpuDescriptor;

void cpu_allgather(void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, Communicator* communicator);

#endif