#ifndef __CUDA_REDUCE_H__
#define __CUDA_REDUCE_H__

#include "operators.h"
#include "ops/reduce/reduce.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include "../../../devices/cuda/common_cuda.h"

typedef struct ReduceCudaDescriptor {
    Device device;
} ReduceCudaDescriptor;

void nv_gpu_reduce(void* sendbuff, void* recvbuff, int count, CCLDatatype datatype, CCLOp op, int root, Communicator* communicator, Stream* stream);

#endif