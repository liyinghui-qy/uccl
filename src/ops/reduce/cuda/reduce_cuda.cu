#include "reduce_cuda.cuh"

void nv_gpu_reduce(void* sendbuff, void* recvbuff, int count, CCLDatatype datatype, CCLOp op, int root, Communicator* communicator, Stream* stream) {
    ncclDataType_t datatype_cuda = ccl_to_cuda_datatype(datatype);
    ncclRedOp_t op_cuda = ccl_to_cuda_op(op);
    ncclComm_t* comm = (ncclComm_t*) communicator;
    cudaStream_t* cudaStream = (cudaStream_t*) stream;
    ncclReduce(sendbuff, recvbuff, count, datatype_cuda, op_cuda, root, *comm, *cudaStream);
}