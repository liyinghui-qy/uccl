#ifndef ALLTOALL_H
#define ALLTOALL_H

#include "../../export.h"
#include "../../operators.h"
#include "../communicator/communicator.h"

typedef struct AlltoallDescriptor AlltoallDescriptor;
typedef struct Stream Stream;

__C __export void *createAlltoallDescriptor(Device device, void *config);
__C __export void destroyAlltoallDescriptor(AlltoallDescriptor *descriptor);
__C __export void Alltoallreduce(AlltoallDescriptor *descriptor, void* sendbuff, void* recvbuff, int count, int datatype, int op, Communicator* communicator, Stream* stream);

#endif