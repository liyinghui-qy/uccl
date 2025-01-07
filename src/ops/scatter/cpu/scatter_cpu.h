#ifndef __CPU_BROADCAST_H__
#define __CPU_BROADCAST_H__

#include "ops/scatter/scatter.h"

typedef struct ScatterCpuDescriptor {
    Device device;
} ScatterCpuDescriptor;

void cpu_scatter(void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, int root, Communicator* communicator);

#endif