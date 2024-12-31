#include "../utils.h"
#include "ops/gather/gather.h"

#ifdef ENABLE_CPU
#include "cpu/gather_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
//#include "cuda/gather_cuda.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/gather_cnnl.h"
#endif

struct GatherDescriptor {
    Device device;
};

__C __export void *createGatherDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (GatherDescriptor *) (new GatherCpuDescriptor{device});
#endif
/*
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (GatherDescriptor *) (new GatherCudaDescriptor(device));
        }
#endif
*/
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (GatherDescriptor *) (new GatherBangDescriptor(device));
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C __export void destroyGatherDescriptor(GatherDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (GatherCpuDescriptor *) (descriptor);
            break;
#endif
/*
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (GatherCudaDescriptor *) (descriptor);
            break;
#endif
*/
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (GatherDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}


__C __export void Gather(GatherDescriptor *descriptor, void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, int root, Communicator* communicator, Stream* stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            cpu_gather(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, root, communicator);
            break;
#endif
/*
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            nv_gpu_gather(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, root, communicator, stream);
            break;
#endif
*/
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            cnnl_gater(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, root, communicator, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}