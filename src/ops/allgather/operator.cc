#include "../utils.h"
#include "ops/allgather/allgather.h"

#ifdef ENABLE_CPU
#include "cpu/allgather_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
//#include "cuda/allgather_cuda.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/allgather_cnnl.h"
#endif

struct AllgatherDescriptor {
    Device device;
};

__C __export void *createAllgatherDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (AllgatherDescriptor *) (new AllgatherCpuDescriptor{device});
#endif
/*
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (AllgatherDescriptor *) (new AllgatherCudaDescriptor(device));
        }
#endif
*/
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (AllgatherDescriptor *) (new AllgatherBangDescriptor(device));
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C __export void destroyAllgatherDescriptor(AllgatherDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (AllgatherCpuDescriptor *) (descriptor);
            break;
#endif
/*
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (AllgatherCudaDescriptor *) (descriptor);
            break;
#endif
*/
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (AllgatherDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}


__C __export void Allgather(AllgatherDescriptor *descriptor, void* sendbuff, int send_count, CCLDatatype send_datatype, void* recvbuff, int recv_count, CCLDatatype recv_datatype, Communicator* communicator, Stream* stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            cpu_allgather(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, communicator);
            break;
#endif
/*
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            nv_gpu_allgather(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, communicator, stream);
            break;
#endif
*/
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            cnnl_allgater(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, communicator, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}