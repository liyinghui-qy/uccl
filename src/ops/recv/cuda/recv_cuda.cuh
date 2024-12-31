#ifndef __CUDA_RECV_H__
#define __CUDA_RECV_H__

#include "operators.h"
#include "ops/recv/recv.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include "../../../devices/cuda/common_cuda.h"

typedef struct RecvCudaDescriptor {
    Device device;
} RecvCudaDescriptor;

void nv_gpu_recv(void* recvbuff, int count, CCLDatatype datatype, int peer, Communicator* communicator, Status* status, Stream* stream);

#endif