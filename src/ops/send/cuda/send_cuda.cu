#include "send_cuda.cuh"
#include <iostream>

#define NCCLCHECK(cmd) do {                         \
    ncclResult_t res = cmd;                         \
    if (res != ncclSuccess) {                       \
        std::cerr << "NCCL error: " << ncclGetErrorString(res) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

void nv_gpu_send(void* sendbuff, int count, CCLDatatype datatype, int peer, Communicator* communicator, Stream* stream) {
    ncclDataType_t datatype_cuda = ccl_to_cuda_datatype(datatype);
    ncclComm_t comm = (ncclComm_t) communicator->comm;
    cudaStream_t* cudaStream = (cudaStream_t*) stream;
    if (stream == nullptr) {
        NCCLCHECK(ncclSend(sendbuff, count, datatype_cuda, peer, comm, 0));
    }
    else {
        NCCLCHECK(ncclSend(sendbuff, count, datatype_cuda, peer, comm, *cudaStream));
    }
    
}