#ifndef RECV_H
#define RECV_H

#include "../../export.h"
#include "../../operators.h"
#include "../communicator/communicator.h"

typedef struct RecvDescriptor RecvDescriptor;
typedef struct Status Status;

__C __export void *createRecvDescriptor(Device device, void *config);
__C __export void destroyRecvDescriptor(RecvDescriptor *descriptor);
__C __export void Recv(RecvDescriptor *descriptor, void* recvbuff, int count, int datatype, int peer, Communicator* communicator, Status* status);

#endif