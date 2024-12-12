#ifndef __CPU_SEND_H__
#define __CPU_SEND_H__

#include "operators.h"
#include "ops/send/send.h"

typedef struct SendCpuDescriptor {
    Device device;
} SendCpuDescriptor;

void cpu_send(void* sendbuff, int count, int datatype, int peer, Communicator* communicator);

#endif