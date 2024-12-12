#ifndef GATHER_H
#define GATHER_H

#include "../../export.h"
#include "../../operators.h"
#include "../communicator/communicator.h"

typedef struct GatherDescriptor GatherDescriptor;
typedef struct Stream Stream;

__C __export void *createGatherDescriptor(Device device, void *config);
__C __export void destroyGatherDescriptor(GatherDescriptor *descriptor);
__C __export void Gather(GatherDescriptor *descriptor, void* sendbuff, int send_count, int send_datatype, void* recvbuff, int recv_count, int recv_datatype, int root, Communicator* communicator, Stream* stream);

#endif