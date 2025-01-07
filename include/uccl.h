#ifndef __UCCL_H__
#define __UCCL_H__

#include "data_type.h"
#include "device.h"

typedef enum DeviceEnum Device;

struct Communicator {
    Device deviceType;
    unsigned int deviceID; // the actual device ID, not rank number
    void *comm;   // the actual communication object
};

struct Stream {
    Device deviceType;
    unsigned int deviceID;
    void* stream;
};

#endif// __OPERATORS_H__
