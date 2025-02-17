#ifndef __COMMON_CUDA_H__
#define __COMMON_CUDA_H__

#include <nccl.h>
#include <cuda_runtime.h>
#include "data_type.h"

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WARP_PER_BLOCK 32
#define WARP_SIZE 32

ncclDataType_t ccl_to_cuda_datatype(CCLDatatype datatype);

ncclRedOp_t ccl_to_cuda_op(CCLOp op);

#endif // __COMMON_CUDA_H__
