#include "recv_cuda.cuh"

void nv_gpu_recv(void* recvbuff, int count, CCLDatatype datatype, int peer, Communicator* communicator, Status* status, Stream* stream) {
    ncclDataType_t datatype_cuda = ccl_to_cuda_datatype(datatype);
    ncclComm_t* comm = (ncclComm_t*) communicator;
    cudaStream_t* cudaStream = (cudaStream_t*) stream;
    ncclRecv(recvbuff, count, datatype_cuda, peer, *comm, *cudaStream);
}
