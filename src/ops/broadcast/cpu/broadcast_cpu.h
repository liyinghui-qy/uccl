#ifndef __CPU_BROADCAST_H__
#define __CPU_BROADCAST_H__

#include "operators.h"
#include "ops/broadcast/broadcast.h"

typedef struct BroadcastCpuDescriptor {
    Device device;
} BroadcastCpuDescriptor;

void cpu_broadcast(void* buff, int count, CCLDatatype datatype, int root, Communicator* communicator);

#endif