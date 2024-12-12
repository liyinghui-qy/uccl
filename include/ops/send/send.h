#ifndef SEND_H
#define SEND_H

#include "../../export.h"
#include "../../operators.h"
#include "../communicator/communicator.h"

typedef struct SendDescriptor SendDescriptor;

__C __export void *createSendDescriptor(Device device, void *config);
__C __export void destroySendDescriptor(SendDescriptor *descriptor);
__C __export void Send(SendDescriptor *descriptor, void* sendbuff, int count, int datatype, int peer, Communicator* communicator);

#endif