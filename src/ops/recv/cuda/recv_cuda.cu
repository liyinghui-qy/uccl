#include "recv_cuda.cuh"
#include <iostream>

#define NCCLCHECK(cmd) do {                         \
    ncclResult_t res = cmd;                         \
    if (res != ncclSuccess) {                       \
        std::cerr << "NCCL error: " << ncclGetErrorString(res) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

void nv_gpu_recv(void* recvbuff, int count, CCLDatatype datatype, int peer, Communicator* communicator, Status* status, Stream* stream) {
    ncclDataType_t datatype_cuda = ccl_to_cuda_datatype(datatype);
    ncclComm_t* comm = (ncclComm_t*) communicator;
    cudaStream_t* cudaStream = (cudaStream_t*) stream;
    NCCLCHECK(ncclRecv(recvbuff, count, datatype_cuda, peer, *comm, *cudaStream));
}
