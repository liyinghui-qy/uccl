#ifndef RECV_H
#define RECV_H

#include "../../export.h"
#include "../../operators.h"
#include "../communicator/communicator.h"
#include "../../data_type.h"

typedef struct RecvDescriptor RecvDescriptor;
typedef struct Status Status;
typedef struct Stream Stream;

__C __export void *createRecvDescriptor(Device device, void *config);
__C __export void destroyRecvDescriptor(RecvDescriptor *descriptor);
__C __export void Recv(RecvDescriptor *descriptor, void* recvbuff, int count, CCLDatatype datatype, int peer, Communicator* communicator, Status* status, Stream* stream);

#endif