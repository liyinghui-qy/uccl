#ifndef __CPU_REDUCE_H__
#define __CPU_REDUCE_H__

#include "operators.h"
#include "ops/reduce/reduce.h"

typedef struct ReduceCpuDescriptor {
    Device device;
} ReduceCpuDescriptor;

void cpu_reduce(void* sendbuff, void* recvbuff, int count, int datatype, int op, int root, Communicator* communicator);

#endif