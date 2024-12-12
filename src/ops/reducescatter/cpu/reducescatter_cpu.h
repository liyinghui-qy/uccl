#ifndef __CPU_REDUCESCATTER_H__
#define __CPU_REDUCESCATTER_H__

#include "operators.h"
#include "ops/reducescatter/reducescatter.h"

typedef struct ReducescatterCpuDescriptor {
    Device device;
} ReducescatterCpuDescriptor;

void cpu_reducescatter(void* sendbuff, void* recvbuff, int* recvcounts, int datatype, int op, Communicator* communicator);

#endif