#ifndef __CUDA_ALLGATHER_H__
#define __CUDA_ALLGATHER_H__

#include "operators.h"
#include "ops/allgather/allgather.h"

typedef struct AllgatherCudaDescriptor {
    Device device;
} AllgatherCudaDescriptor;

void nv_gpu_allgather(void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, Communicator* communicator, Stream* stream);

#endif

