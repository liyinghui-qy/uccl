#include "../utils.h"
#include "ops/reduce/reduce.h"

#ifdef ENABLE_CPU
#include "cpu/reduce_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/reduce_cuda.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/reduce_cnnl.h"
#endif

struct ReduceDescriptor {
    Device device;
};

__C __export void *createReduceDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (ReduceDescriptor *) (new ReduceCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (ReduceDescriptor *) (new ReduceCudaDescriptor(device));
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (ReduceDescriptor *) (new ReduceBangDescriptor(device));
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C __export void destroyReduceDescriptor(ReduceDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (ReduceCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (ReduceCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (ReduceDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}


__C __export void Reduce(ReduceDescriptor *descriptor, void* sendbuff, void* recvbuff, int count, CCLDatatype datatype, CCLOp op, int root, Communicator* communicator, Stream* stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            cpu_reduce(sendbuff, recvbuff, count, datatype, op, root, communicator);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            nv_gpu_reduce(sendbuff, recvbuff, count, datatype, op, root, communicator, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            cnnl_reduce(sendbuff, recvbuff, count, datatype, op, root, communicator, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}