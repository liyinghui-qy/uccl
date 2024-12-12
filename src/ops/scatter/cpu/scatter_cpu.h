#ifndef __CPU_BROADCAST_H__
#define __CPU_BROADCAST_H__

#include "operators.h"
#include "ops/scatter/scatter.h"

typedef struct ScatterCpuDescriptor {
    Device device;
} ScatterCpuDescriptor;

void cpu_scatter(void* sendbuff, int send_count, int send_datatype, void* recvbuff, int recv_count, int recv_datatype, int root, Communicator* communicator);

#endif