#ifndef __CUDA_ALLREDUCE_H__
#define __CUDA_ALLREDUCE_H__


#include "ops/allreduce/allreduce.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include "../../../devices/cuda/common_cuda.h"

typedef struct AllreduceCudaDescriptor {
    Device device;
} AllreduceCudaDescriptor;

void nv_gpu_allreduce(void* sendbuff, void* recvbuff, int count, CCLDatatype datatype, CCLOp op, Communicator* communicator, Stream* stream);

#endif