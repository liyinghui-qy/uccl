#include <vector>
#include <iostream>
#include "communicator_cuda.cuh"
#include "../../../devices/cuda/common_cuda.h"

#define CUDACHECK(cmd) do {                         \
    cudaError_t err = cmd;                          \
    if (err != cudaSuccess) {                       \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

#define NCCLCHECK(cmd) do {                         \
    ncclResult_t res = cmd;                         \
    if (res != ncclSuccess) {                       \
        std::cerr << "NCCL error: " << ncclGetErrorString(res) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

void communicator_nv_gpu_init(CommunicatorDescriptor* descriptor, Communicator* comm) {
    CommunicatorCudaDescriptor* dcp = (CommunicatorCudaDescriptor *)descriptor;
    NCCLCHECK(ncclCommInitAll((ncclComm_t *)comm, dcp->num_gpus, dcp->devices));
}

Communicator* get_nv_gpu_communicator(CommunicatorDescriptor* descriptor) {
    CommunicatorCudaDescriptor* dcp = (CommunicatorCudaDescriptor *)descriptor;
    std::vector<ncclComm_t>* comms = new std::vector<ncclComm_t>(dcp->num_gpus);
    return (Communicator *)((*comms).data());
}

void get_nv_gpu_comm_size(Communicator* comm, int* size) {
    ncclComm_t* comm_nccl = (ncclComm_t *)comm;
    NCCLCHECK(ncclCommCount(*comm_nccl, size));
}

void get_nv_gpu_comm_rank(Communicator* comm, int* rank) {
    ncclComm_t* comm_nccl = (ncclComm_t *)comm;
    NCCLCHECK(ncclCommUserRank(*comm_nccl, rank));
}