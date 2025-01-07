#ifndef __CPU_RECV_H__
#define __CPU_RECV_H__

#include "ops/recv/recv.h"

typedef struct RecvCpuDescriptor {
    Device device;
} RecvCpuDescriptor;

void cpu_recv(void* recvbuff, int count, CCLDatatype datatype, int peer, Communicator* communicator, Status* status);

#endif