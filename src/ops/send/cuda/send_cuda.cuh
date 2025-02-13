#ifndef __CUDA_SEND_H__
#define __CUDA_SEND_H__


#include "ops/send/send.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include "../../../devices/cuda/common_cuda.h"

typedef struct SendCudaDescriptor {
    Device device;
} SendCudaDescriptor;

void nv_gpu_send(void* sendbuff, int count, CCLDatatype datatype, int peer, Communicator* communicator, Stream* stream);

#endif