#include "../utils.h"
#include "ops/communicator/communicator.h"

#ifdef ENABLE_CPU
#include "cpu/communicator_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/communicator_cuda.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/communicator_cnnl.h"
#endif

struct CommunicatorDescriptor {
    Device device;
};

__C CommunicatorDescriptor *createCommunicatorDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (CommunicatorDescriptor *) (new CommunicatorCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (CommunicatorDescriptor *) (new ComunicatorCudaDescriptor(device));
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return (CommunicatorDescriptor *) (new CommunicatorBangDescriptor(device));
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C void destroyCommunicatorDescriptor(CommunicatorDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (CommunicatorCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (CommunicatorCudaDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (CommunicatorBangDescriptor *) (descriptor);
            break;
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C void communicator_init(CommunicatorDescriptor* descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            communicator_cpu_init();
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            communicator_nv_gpu_init();
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            communicator_cnnl_init();
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C Communicator *get_communicator(CommunicatorDescriptor* descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu: {
            return get_cpu_communicator();
        }
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return get_nv_gpu_communicator();
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return get_cnnl_communicator();
        }
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C void get_comm_size(CommunicatorDescriptor* descriptor, Communicator* comm, int* size) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            get_cpu_commm_size(comm, size);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            get_nv_gpu_comm_size(comm, size);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            get_cnnl_comm_size(comm, size);
            break;
#endif
        default:
            printf("device type is %d\n", descriptor->device);
            PANIC(UnsupportedDevice);
    }
}

__C void get_comm_rank(CommunicatorDescriptor* descriptor, Communicator* comm, int* rank) {
        switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            get_cpu_commm_rank(comm, rank);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            get_nv_gpu_comm_size(comm, rank);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            get_cnnl_comm_size(comm, rank);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}