#ifndef REDUCESCATTER_H
#define REDUCESCATTER_H

#include "../../export.h"
#include "../../operators.h"
#include "../communicator/communicator.h"

typedef struct ReducescatterDescriptor ReducescatterDescriptor;
typedef struct Stream Stream;

__C __export void *createReducescatterDescriptor(Device device, void *config);
__C __export void destroyReducescatterDescriptor(ReducescatterDescriptor *descriptor);
__C __export void Reducescatter(ReducescatterDescriptor *descriptor, void* sendbuff, void* recvbuff, int* recvcounts, CCLDatatype datatype, CCLOp op, Communicator* communicator, Stream* stream);

#endif