#ifndef SCATTER_H
#define SCATTER_H

#include "../../export.h"
#include "../../operators.h"
#include "../communicator/communicator.h"
#include "../../data_type.h"

typedef struct ScatterDescriptor ScatterDescriptor;
typedef struct Stream Stream;

__C __export void *createScatterDescriptor(Device device, void *config);
__C __export void destroyScatterDescriptor(ScatterDescriptor *descriptor);
__C __export void Scatter(ScatterDescriptor *descriptor, void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, int root, Communicator* communicator, Stream* stream);

#endif