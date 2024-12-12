#include "../utils.h"
#include "ops/allreduce/allreduce.h"

#ifdef ENABLE_CPU
#include "cpu/allreduce_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/allreduce_cuda.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/allreduce_cnnl.h"
#endif

struct AllreduceDescriptor {
    Device device;
};

__C __export void *createAllreduceDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (AllreduceDescriptor *) (new AllreduceCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (AllreduceDescriptor *) (new AllreduceCudaDescriptor(device));
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (AllreduceDescriptor *) (new AllreduceBangDescriptor(device));
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C __export void destroyAllreduceDescriptor(AllreduceDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (AllreduceCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (AllreduceCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (AllreduceDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}


__C __export void Allreduce(AllreduceDescriptor *descriptor, void* sendbuff, void* recvbuff, int count, int datatype, int op, Communicator* communicator, Stream* stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            cpu_allreduce(sendbuff, recvbuff, count, datatype, op, communicator);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            nv_gpu_allreduce(sendbuff, recvbuff, count, datatype, op, communicator, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            cnnl_allreduce(sendbuff, recvbuff, count, datatype, op, communicator, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}