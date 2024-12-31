#include "send_cuda.cuh"

void nv_gpu_send(void* sendbuff, int count, CCLDatatype datatype, int peer, Communicator* communicator, Stream* stream) {
    ncclDataType_t datatype_cuda = ccl_to_cuda_datatype(datatype);
    ncclComm_t* comm = (ncclComm_t*) communicator;
    cudaStream_t* cudaStream = (cudaStream_t*) stream;
    ncclSend(sendbuff, count, datatype_cuda, peer, *comm, *cudaStream);
}