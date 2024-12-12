#include "../utils.h"
#include "ops/broadcast/broadcast.h"

#ifdef ENABLE_CPU
#include "cpu/broadcast_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/broadcast_cuda.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/broadcast_cnnl.h"
#endif

struct BroadcastDescriptor {
    Device device;
};

__C __export void *createBroadcastDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (BroadcastDescriptor *) (new BroadcastCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (BroadcastDescriptor *) (new BroadcastCudaDescriptor(device));
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (BroadcastDescriptor *) (new BroadcastBangDescriptor(device));
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C __export void destroyBroadcastDescriptor(BroadcastDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (BroadcastCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (BroadcastCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (BroadcastDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}


__C __export void Broadcast(BroadcastDescriptor *descriptor, void* buff, int count, int datatype, int root, Communicator* communicator, Stream* stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            cpu_broadcast(buff, count, datatype, root, communicator);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            nv_gpu_broadcast(buff, count, datatype, root, communicator, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            cnnl_recv(buff, count, datatype, root, communicator, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}