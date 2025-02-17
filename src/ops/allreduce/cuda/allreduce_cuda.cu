#include "allreduce_cuda.cuh"
#include <iostream>

#define NCCLCHECK(cmd) do {                         \
    ncclResult_t res = cmd;                         \
    if (res != ncclSuccess) {                       \
        std::cerr << "NCCL error: " << ncclGetErrorString(res) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

void nv_gpu_allreduce(void* sendbuff, void* recvbuff, int count, CCLDatatype datatype, CCLOp op, Communicator* communicator, Stream* stream) {
    ncclDataType_t datatype_cuda = ccl_to_cuda_datatype(datatype);
    ncclRedOp_t op_cuda = ccl_to_cuda_op(op);
    ncclComm_t comm = (ncclComm_t) communicator->comm;
    cudaStream_t* cudaStream = (cudaStream_t*) stream;

    cudaSetDevice(communicator->deviceID);
    if (cudaStream == nullptr) {
        NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, datatype_cuda, op_cuda, comm, 0));
    }
    else {
        NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, datatype_cuda, op_cuda, comm, *cudaStream));
    }
}