#include "allgather_cuda.cuh"
#include "../../../devices/cuda/common_cuda.h"


void nv_gpu_allgather(void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, Communicator* communicator, Stream* stream) {
    ncclDataType_t datatype_cuda = ccl_to_cuda_datatype(send_datatype);
    ncclComm_t* comm = (ncclComm_t*) communicator;
    cudaStream_t* cudaStream = (cudaStream_t*) stream;
    ncclAllGather(sendbuff, recvbuff, send_count, datatype_cuda, *comm, *cudaStream);
}