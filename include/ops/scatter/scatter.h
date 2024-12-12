#ifndef SCATTER_H
#define SCATTER_H

#include "../../export.h"
#include "../../operators.h"
#include "../communicator/communicator.h"

typedef struct ScatterDescriptor ScatterDescriptor;
typedef struct Stream Stream;

__C __export void *createScatterDescriptor(Device device, void *config);
__C __export void destroyScatterDescriptor(ScatterDescriptor *descriptor);
__C __export void Scatter(ScatterDescriptor *descriptor, void* sendbuff, int send_count, int send_datatype, void* recvbuff, int recv_count, int recv_datatype, int root, Communicator* communicator, Stream* stream);

#endif