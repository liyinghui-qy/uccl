#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include "../../export.h"
#include "../../operators.h"

typedef struct CommunicatorDescriptor CommunicatorDescriptor;
struct Communicator {
    Device deviceType;
    unsigned int deviceID; // the actual device ID, not rank number
    void *comm;   // the actual communication object
};

__C __export CommunicatorDescriptor *createCommunicatorDescriptor(Device, void *config);
__C __export void destroyCommunicatorDescriptor(CommunicatorDescriptor *descriptor);
__C __export void communicator_init(CommunicatorDescriptor* descriptor, Communicator* comm);
__C __export Communicator *get_communicator(CommunicatorDescriptor* descriptor);
__C __export void get_comm_size(CommunicatorDescriptor* descriptor, Communicator* comm, int* size);
__C __export void get_comm_rank(CommunicatorDescriptor* descriptor, Communicator* comm, int* rank);

#endif