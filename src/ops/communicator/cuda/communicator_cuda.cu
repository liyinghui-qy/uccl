#include <vector>
#include <iostream>
#include "communicator_cuda.cuh"
#include "../../../devices/cuda/common_cuda.h"

#define NCCLCHECK(cmd) do {                         \
    ncclResult_t res = cmd;                         \
    if (res != ncclSuccess) {                       \
        std::cerr << "NCCL error: " << ncclGetErrorString(res) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

//接口不用
void communicator_nv_gpu_init(CommunicatorDescriptor* descriptor, Communicator* comms) {
    CommunicatorCudaDescriptor* dcp = (CommunicatorCudaDescriptor *)descriptor;
    NCCLCHECK(ncclCommInitAll((ncclComm_t *)comms, dcp->num_gpus, dcp->devices));
}

Communicator* get_nv_gpu_communicator(CommunicatorDescriptor* descriptor) {
    CommunicatorCudaDescriptor* dcp = (CommunicatorCudaDescriptor *)descriptor;
    std::vector<ncclComm_t> comms(dcp->num_gpus);
    NCCLCHECK(ncclCommInitAll(comms.data(), dcp->num_gpus, dcp->devices));
    Communicator* communicators = new Communicator[dcp->num_gpus];
    for(int i = 0; i < dcp->num_gpus; i++) {
        communicators[i].deviceID = dcp->devices[i];
        communicators[i].deviceType = dcp->device;
        communicators[i].comm = (void*)comms[i];
    }
    return communicators;
}

void get_nv_gpu_comm_size(Communicator* communicator, int* size) {
    ncclComm_t comm_nccl = (ncclComm_t)communicator->comm;
    NCCLCHECK(ncclCommCount(comm_nccl, size));
}

void get_nv_gpu_comm_rank(Communicator* communicator, int* rank) {
    ncclComm_t comm_nccl = (ncclComm_t)communicator->comm;
    NCCLCHECK(ncclCommUserRank(comm_nccl, rank));
}