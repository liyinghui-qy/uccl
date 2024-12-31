#include "reducescatter_cuda.cuh"

void nv_gpu_reducescatter(void* sendbuff, void* recvbuff, int* recvcounts, CCLDatatype datatype, CCLOp op, Communicator* communicator, Stream* stream) {
    ncclDataType_t datatype_cuda = ccl_to_cuda_datatype(datatype);
    ncclRedOp_t op_cuda = ccl_to_cuda_op(op);
    ncclComm_t* comm = (ncclComm_t*) communicator;
    cudaStream_t* cudaStream = (cudaStream_t*) stream;
    ncclReduceScatter(sendbuff, recvbuff, *recvcounts, datatype_cuda, op_cuda, *comm, *cudaStream);
}
