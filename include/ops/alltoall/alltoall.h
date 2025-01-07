#ifndef ALLTOALL_H
#define ALLTOALL_H

#include "../../export.h"
#include "../../uccl.h"
#include "../communicator/communicator.h"
#include "../../data_type.h"

typedef struct AlltoallDescriptor AlltoallDescriptor;
typedef struct Stream Stream;

__C __export void *createAlltoallDescriptor(Device device, void *config);
__C __export void destroyAlltoallDescriptor(AlltoallDescriptor *descriptor);
__C __export void Alltoall(AlltoallDescriptor *descriptor, void* sendbuff, int sendcount, CCLDatatype send_datatype, void* recvbuff, int count, CCLDatatype recv_datatype, Communicator* communicator, Stream* stream);

#endif