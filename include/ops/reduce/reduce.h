#ifndef REDUCE_H
#define REDUCE_H

#include "../../export.h"
#include "../../uccl.h"
#include "../communicator/communicator.h"

typedef struct ReduceDescriptor ReduceDescriptor;
typedef struct Stream Stream;

__C __export void *createReduceDescriptor(Device device, void *config);
__C __export void destroyReduceDescriptor(ReduceDescriptor *descriptor);
__C __export void Reduce(ReduceDescriptor *descriptor, void* sendbuff, void* recvbuff, int count, CCLDatatype datatype, CCLOp op, int root, Communicator* communicator, Stream* stream);

#endif