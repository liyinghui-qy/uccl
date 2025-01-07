#ifndef ALLGATHER_H
#define ALLGATHER_H

#include "../../export.h"
#include "../../uccl.h"
#include "../communicator/communicator.h"
#include "../../data_type.h"

typedef struct AllgatherDescriptor AllgatherDescriptor;
typedef struct Stream Stream;

__C __export void *createAllgatherDescriptor(Device device, void *config);
__C __export void destroyAllgatherDescriptor(AllgatherDescriptor *descriptor);
__C __export void Allgather(AllgatherDescriptor *descriptor, void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, Communicator* communicator, Stream* stream);

#endif