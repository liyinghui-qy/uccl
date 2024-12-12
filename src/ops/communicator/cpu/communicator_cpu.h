#ifndef __CPU_COMMUNICATOR_H__
#define __CPU_COMMUNICATOR_H__

#include "operators.h"
#include "ops/communicator/communicator.h"
typedef struct CommunicatorCpuDescriptor {
    Device device;
} CommunicatorCpuDescriptor;

void communicator_cpu_init();

Communicator* get_cpu_communicator();

void get_cpu_commm_size(Communicator* comm, int* size);

void get_cpu_commm_rank(Communicator* comm, int* rank);

#endif