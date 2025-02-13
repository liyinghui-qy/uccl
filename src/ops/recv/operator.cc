#include "../utils.h"
#include "ops/recv/recv.h"

#ifdef ENABLE_CPU
#include "cpu/recv_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/recv_cuda.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/recv_cnnl.h"
#endif

struct RecvDescriptor {
    Device device;
};

__C __export void *createRecvDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (RecvDescriptor *) (new RecvCpuDescriptor{device});
#endif

#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (RecvDescriptor *) (new RecvCudaDescriptor{device});
        }
#endif

#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (RecvDescriptor *) (new RecvBangDescriptor(device));
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C __export void destroyRecvDescriptor(RecvDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (RecvCpuDescriptor *) (descriptor);
            break;
#endif

#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (RecvCudaDescriptor *) (descriptor);
            break;
#endif

#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (RecvBangDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}


__C __export void Recv(RecvDescriptor *descriptor, void* recvbuff, int count, CCLDatatype datatype, int peer, Communicator* communicator, Status* status, Stream* stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            cpu_recv(recvbuff, count, datatype, peer, communicator, status);
            break;
#endif

#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            nv_gpu_recv(recvbuff, count, datatype, peer, communicator, status, stream);
            break;
#endif

#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            cnnl_recv(recvbuff, count, datatype, peer, communicator, status);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}