#include "../utils.h"
#include "ops/reducescatter/reducescatter.h"

#ifdef ENABLE_CPU
#include "cpu/reducescatter_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/reducescatter_cuda.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/reducescatter_cnnl.h"
#endif

struct ReducescatterDescriptor {
    Device device;
};

__C __export void *createReducescatterDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (ReducescatterDescriptor *) (new ReducescatterCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (ReducescatterDescriptor *) (new ReducescatterCudaDescriptor(device));
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (ReducescatterDescriptor *) (new ReducescatterBangDescriptor(device));
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C __export void destroyReducescatterDescriptor(ReducescatterDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (ReducescatterCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (ReducescatterCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (ReducescatterDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}


__C __export void Reducescatter(ReducescatterDescriptor *descriptor, void* sendbuff, void* recvbuff, int* recvcounts, int datatype, int op, Communicator* communicator, Stream* stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            cpu_reducescatter(sendbuff, recvbuff, recvcounts, datatype, op, communicator);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            nv_gpu_reducescatter(sendbuff, recvbuff, recvcounts, datatype, op, communicator, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            cnnl_reducescatter(sendbuff, recvbuff, recvcounts, datatype, op, communicator, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}