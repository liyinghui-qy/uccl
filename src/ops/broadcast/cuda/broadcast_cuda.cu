#include "broadcast_cuda.cuh"
#include <iostream>

#define NCCLCHECK(cmd) do {                         \
    ncclResult_t res = cmd;                         \
    if (res != ncclSuccess) {                       \
        std::cerr << "NCCL error: " << ncclGetErrorString(res) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

void nv_gpu_broadcast(void* buff, int count, CCLDatatype datatype, int root, Communicator* communicator, Stream* stream) {
    ncclDataType_t datatype_cuda = ccl_to_cuda_datatype(datatype);
    ncclComm_t* comm = (ncclComm_t*) communicator;
    cudaStream_t* cudaStream = (cudaStream_t*) stream;
    NCCLCHECK(ncclAllGather(buff, buff, count, datatype_cuda, *comm, *cudaStream));
}