#include "broadcast_cuda.cuh"

void nv_gpu_broadcast(void* buff, int count, CCLDatatype datatype, int root, Communicator* communicator, Stream* stream) {
    ncclDataType_t datatype_cuda = ccl_to_cuda_datatype(datatype);
    ncclComm_t* comm = (ncclComm_t*) communicator;
    cudaStream_t* cudaStream = (cudaStream_t*) stream;
    ncclAllGather(buff, buff, count, datatype_cuda, *comm, *cudaStream);
}