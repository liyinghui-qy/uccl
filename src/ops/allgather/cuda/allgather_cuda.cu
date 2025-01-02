#include "allgather_cuda.cuh"
#include <iostream>

#define NCCLCHECK(cmd) do {                         \
    ncclResult_t res = cmd;                         \
    if (res != ncclSuccess) {                       \
        std::cerr << "NCCL error: " << ncclGetErrorString(res) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

void nv_gpu_allgather(void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, Communicator* communicator, Stream* stream) {
    ncclDataType_t datatype_cuda = ccl_to_cuda_datatype(send_datatype);
    ncclComm_t* comm = (ncclComm_t*) communicator;
    cudaStream_t* cudaStream = (cudaStream_t*) stream;
    NCCLCHECK(ncclAllGather(sendbuff, recvbuff, send_count, datatype_cuda, *comm, *cudaStream));
}