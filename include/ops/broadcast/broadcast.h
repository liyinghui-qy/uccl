#ifndef BROADCAST_H
#define BROADCAST_H

#include "../../export.h"
#include "../../uccl.h"
#include "../communicator/communicator.h"
#include "../../data_type.h"

typedef struct BroadcastDescriptor BroadcastDescriptor;
typedef struct Stream Stream;

__C __export void *createBroadcastDescriptor(Device device, void *config);
__C __export void destroyBroadcastDescriptor(BroadcastDescriptor *descriptor);
__C __export void Broadcast(BroadcastDescriptor *descriptor, void* buff, int count, CCLDatatype datatype, int root, Communicator* communicator, Stream* stream);

#endif