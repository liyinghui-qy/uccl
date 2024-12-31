#include "../utils.h"
#include "ops/scatter/scatter.h"

#ifdef ENABLE_CPU
#include "cpu/scatter_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
//#include "cuda/scatter_cuda.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/scatter_cnnl.h"
#endif

struct ScatterDescriptor {
    Device device;
};

__C __export void *createScatterDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (ScatterDescriptor *) (new ScatterCpuDescriptor{device});
#endif
/*
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (ScatterDescriptor *) (new ScatterCudaDescriptor(device));
        }
#endif
*/
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (ScatterDescriptor *) (new ScatterBangDescriptor(device));
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C __export void destroyScatterDescriptor(ScatterDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (ScatterCpuDescriptor *) (descriptor);
            break;
#endif
/*
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (ScatterCudaDescriptor *) (descriptor);
            break;
#endif
*/
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (ScatterDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}


__C __export void Scatter(ScatterDescriptor *descriptor, void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, int root, Communicator* communicator, Stream* stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            cpu_scatter(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, root, communicator);
            break;
#endif
/*
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            nv_gpu_scatter(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, root, communicator, stream);
            break;
#endif
*/
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            cnnl_scatter(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, root, communicator, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}