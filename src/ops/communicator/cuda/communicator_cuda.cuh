#ifndef __CUDA_COMMUNICATOR_H__
#define __CUDA_COMMUNICATOR_H__


#include "ops/communicator/communicator.h"

typedef struct CommunicatorCudaDescriptor {
    Device device;
    const int num_gpus;
    const int* devices;
} CommunicatorCudaDescriptor;

void communicator_nv_gpu_init(CommunicatorDescriptor* descriptor, Communicator* comm);

Communicator* get_nv_gpu_communicator(CommunicatorDescriptor* descriptor);

void get_nv_gpu_comm_size(Communicator* comm, int* size);

void get_nv_gpu_comm_rank(Communicator* comm, int* rank);

#endif