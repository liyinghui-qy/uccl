#ifndef __CUDA_BROADCAST_H__
#define __CUDA_BROADCAST_H__

#include "operators.h"
#include "ops/broadcast/broadcast.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include "../../../devices/cuda/common_cuda.h"

typedef struct BroadcastCudaDescriptor {
    Device device;
} BroadcastCudaDescriptor;

void nv_gpu_broadcast(void* buff, int count, CCLDatatype datatype, int root, Communicator* communicator, Stream* stream);

#endif