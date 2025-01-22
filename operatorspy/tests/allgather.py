import ctypes
from ctypes import *
import sys
import os
import random
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from datatypes import *
from operatorspy import (
    open_lib,
    DeviceEnum,
)
   
class Config(ctypes.Structure):
    _fields_ = [("num_gpus", ctypes.c_int),
                ("devices", ctypes.POINTER(ctypes.c_int))]

class Communicator(ctypes.Structure):
    _fields_ = [
        ("deviceID", ctypes.c_int),
        ("deviceType", ctypes.c_int),
        ("comm", ctypes.c_void_p)
    ]
class CommunicatorDescriptor(Structure):
    _fields_ = [("device", c_int)]

class ScatterDescriptor(Structure):
    _fields_ = [("device", c_int)]

    
def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    descriptor = lib.createCommunicatorDescriptor(device, None)
    lib.communicator_init(descriptor)
    comm = lib.get_communicator(descriptor)
    rank = ctypes.c_int(0)
    lib.get_comm_rank(descriptor, comm, ctypes.byref(rank))
    comm_size = ctypes.c_int(0)
    lib.get_comm_size(descriptor, comm, ctypes.byref(comm_size))
    
    data_size = 4
    send_array_type = ctypes.c_int * (data_size)
    send_data = send_array_type()
    recv_array_type = ctypes.c_int * (data_size * comm_size.value)
    recv_data = recv_array_type()

    for i in range(data_size):
        send_data[i] = random.randint(0, 100)
    python_list = [send_data[i] for i in range(data_size)]
    print("rank", rank.value, "sends data:", python_list)

    lib.Allgather(descriptor, ctypes.byref(send_data), data_size, datatypes.CCL_INT, ctypes.byref(recv_data),
        data_size, datatypes.CCL_INT, comm, None)
    python_list = [recv_data[i] for i in range(data_size * comm_size.value)]
    print("rank", rank.value, "receives data:", python_list)
    lib.destroyCommunicatorDescriptor(descriptor)

def test_gpu(lib):
    device = DeviceEnum.DEVICE_CUDA
    config = Config()
    devices_data = [0, 1]
    config.num_gpus = len(devices_data)
    devices_type = ctypes.c_int * config.num_gpus
    config.devices = devices_type(*devices_data)
    descriptor = lib.createCommunicatorDescriptor(device, ctypes.byref(config))
    comms = lib.get_communicator(descriptor)
    comm_size = ctypes.c_int(0)
    rank = ctypes.c_int(0)
    for i in range(config.num_gpus):
        lib.get_comm_size(descriptor, ctypes.byref(comms[i]), ctypes.byref(comm_size))
        lib.get_comm_rank(descriptor, ctypes.byref(comms[i]), ctypes.byref(rank))
        print("communicator size is:", comm_size.value, "communicator rank is:", rank.value)

    data_size = 4
    send_data_gpu = [None] * config.num_gpus
    recv_data_gpu = [None] * config.num_gpus

    recv_data_host = torch.empty(data_size * comm_size.value, dtype=torch.float32)

    # streams = [cuda.Stream() for _ in range(config.num_gpus)]

    for i, device_id in enumerate(devices_data):
        torch.cuda.set_device(device_id)
        send_data_host = torch.rand(data_size, dtype=torch.float32)
        send_data_gpu[i] = send_data_host.cuda()
        recv_data_gpu[i] = torch.empty(data_size * comm_size.value, dtype=torch.float32).cuda()
        print(f"rank {i} sends data: {send_data_host.tolist()}")

    for i, device_id in enumerate(devices_data):
        torch.cuda.set_device(device_id)
        send_data_gpu_ptr = send_data_gpu[i].data_ptr()
        recv_data_gpu_ptr = recv_data_gpu[i].data_ptr()

        lib.Allgather(
            descriptor,
            send_data_gpu_ptr,
            data_size,
            datatypes.CCL_FLOAT,
            recv_data_gpu_ptr,
            data_size * comm_size.value,
            datatypes.CCL_FLOAT,
            comms[i],
            None
        )

    torch.cuda.synchronize()

    for i, device_id in enumerate(devices_data):
        torch.cuda.set_device(device_id)
        recv_data_host.copy_(recv_data_gpu[i].cpu())
        print(f"rank {i} receives data: {recv_data_host.tolist()}")

    lib.destroyCommunicatorDescriptor(descriptor)

if __name__ == "__main__":
    print("Start test cpu")
    dl = ctypes.cdll.LoadLibrary
    lib = open_lib()
    lib.createCommunicatorDescriptor.restype = ctypes.POINTER(CommunicatorDescriptor)
    lib.createCommunicatorDescriptor.argtypes = [ctypes.c_int, c_void_p]
    lib.communicator_init.restype = c_void_p
    lib.communicator_init.argtypes = [c_void_p]
    lib.get_communicator.restype = ctypes.POINTER(Communicator)
    lib.get_communicator.argtypes = [c_void_p]
    lib.get_comm_rank.restype = c_void_p
    lib.get_comm_rank.argtypes = [c_void_p, c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.get_comm_size.restype = c_void_p
    lib.get_comm_size.argtypes = [c_void_p, c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.Allgather.restype = c_void_p
    lib.Allgather.argtypes = [c_void_p, c_void_p, c_int, c_int, c_void_p, c_int, c_int, ctypes.POINTER(Communicator), c_void_p]
    #print("Start test", flush = True)
    #test_cpu(lib)

    print("Start test gpu", flush = True)
    test_gpu(lib)