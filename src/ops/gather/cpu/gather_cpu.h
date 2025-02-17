#ifndef __CPU_GATHER_H__
#define __CPU_GATHER_H__

#include "ops/gather/gather.h"

typedef struct GatherCpuDescriptor {
    Device device;
} GatherCpuDescriptor;

void cpu_gather(void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, int root, Communicator* communicator);

#endif