#ifndef __CPU_REDUCE_H__
#define __CPU_REDUCE_H__

#include "ops/reduce/reduce.h"

typedef struct ReduceCpuDescriptor {
    Device device;
} ReduceCpuDescriptor;

void cpu_reduce(void* sendbuff, void* recvbuff, int count, CCLDatatype datatype, CCLOp op, int root, Communicator* communicator);

#endif