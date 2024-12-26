#ifndef ALLREDUCE_H
#define ALLREDUCE_H

#include "../../export.h"
#include "../../operators.h"
#include "../communicator/communicator.h"

typedef struct AllreduceDescriptor AllreduceDescriptor;
typedef struct Stream Stream;

__C __export void *createAllreduceDescriptor(Device device, void *config);
__C __export void destroyAllreduceDescriptor(AllreduceDescriptor *descriptor);
__C __export void Allreduce(AllreduceDescriptor *descriptor, void* sendbuff, void* recvbuff, int count, CCLDatatype datatype, CCLOp op, Communicator* communicator, Stream* stream);

#endif