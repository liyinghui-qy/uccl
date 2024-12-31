#ifndef __CUDA_REDUCESCATTER_H__
#define __CUDA_REDUCESCATTER_H__

#include "operators.h"
#include "ops/reducescatter/reducescatter.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include "../../../devices/cuda/common_cuda.h"

typedef struct ReducescatterCudaDescriptor {
    Device device;
} ReducescatterCudaDescriptor;

void nv_gpu_reducescatter(void* sendbuff, void* recvbuff, int* recvcounts, CCLDatatype datatype, CCLOp op, Communicator* communicator, Stream* stream);

#endif