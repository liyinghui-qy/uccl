#include "../utils.h"
#include "ops/send/send.h"

#ifdef ENABLE_CPU
#include "cpu/send_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/send_cuda.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/send_cnnl.h"
#endif

struct SendDescriptor {
    Device device;
};

__C __export void *createSendDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (SendDescriptor *) (new SendCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (SendDescriptor *) (new SendCudaDescriptor(device));
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (SendDescriptor *) (new SendBangDescriptor(device));
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C __export void destroySendDescriptor(SendDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (SendCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (SendCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (SendBangDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}


__C __export void Send(SendDescriptor *descriptor, void* sendbuff, int count, CCLDatatype datatype, int peer, Communicator* communicator) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            cpu_send(sendbuff, count, datatype, peer, communicator);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            nv_gpu_send(sendbuff, count, datatype, peer, communicator);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            cnnl_send(sendbuff, count, datatype, peer, communicator);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}