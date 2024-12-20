#include "../utils.h"
#include "ops/alltoall/alltoall.h"

#ifdef ENABLE_CPU
#include "cpu/alltoall_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/alltoall_cuda.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/alltoall_cnnl.h"
#endif

struct AlltoallDescriptor {
    Device device;
};

__C __export void *createAlltoallDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (AlltoallDescriptor *) (new AlltoallCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (AlltoallDescriptor *) (new AlltoallCudaDescriptor(device));
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (AlltoallDescriptor *) (new AlltoallBangDescriptor(device));
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C __export void destroyAlltoallDescriptor(AlltoallDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (AlltoallCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (AlltoallCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (AlltoallDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}


__C __export void Alltoall(AlltoallDescriptor *descriptor, void* sendbuff, int send_count, int send_datatype, void* recvbuff, int recv_count, int recv_datatype, Communicator* communicator, Stream* stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            cpu_alltoall(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, communicator);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            nv_gpu_alltoall(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, communicator, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            cnnl_alltoall(sendbuff, send_count, send_datatype, recvbuff, recv_count, recv_datatype, communicator, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}