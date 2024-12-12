#ifndef __CPU_ALLREDUCE_H__
#define __CPU_ALLREDUCE_H__

#include "operators.h"
#include "ops/allreduce/allreduce.h"

typedef struct AllreduceCpuDescriptor {
    Device device;
} AllreduceCpuDescriptor;

void cpu_allreduce(void* sendbuff, void* recvbuff, int count, int datatype, int op, Communicator* communicator);

#endif