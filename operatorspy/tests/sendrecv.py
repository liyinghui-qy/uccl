import ctypes
from ctypes import *
import sys
import os
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
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

class SendDescriptor(Structure):
    _fields_ = [("device", c_int)]

class RecvDescriptor(Structure):
    _fields_ = [("device", c_int)]

def test_send(lib, descriptor, torch_device):
    comm = lib.get_communicator(descriptor)
    a = ctypes.c_int(100)
    lib.Send(descriptor, ctypes.byref(a), 1, datatypes.CCL_INT, 1, comm)
    print("Send data", a.value, "to rank 1", flush = True)

def test_recv(lib, descriptor, torch_device):
    comm = lib.get_communicator(descriptor)
    b = ctypes.c_int(0)
    lib.Recv(descriptor, ctypes.byref(b), 1, datatypes.CCL_INT, 0, comm, None)
    print("Recv data", b.value, "from rank 0", flush = True)
    
def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    descriptor = lib.createCommunicatorDescriptor(device, None)
    lib.communicator_init(descriptor)
    comm = lib.get_communicator(descriptor)
    rank = ctypes.c_int(0)
    lib.get_comm_rank(descriptor, comm, ctypes.byref(rank))
    print("current rank is:", rank.value, flush = True)
    if rank.value == 0:
        print("start send", flush = True)
        test_send(lib, descriptor, "cpu")
    else:
        print("start recv", flush = True)
        test_recv(lib, descriptor, "cpu")
    lib.destroyCommunicatorDescriptor(descriptor)

def test_gpu(lib):
    device = DeviceEnum.DEVICE_CUDA
    config = Config()
    devices_data = [0, 1]
    config.num_gpus = len(devices_data)
    devices_type = c_int * config.num_gpus
    config.devices = devices_type(*devices_data)
    descriptor = lib.createCommunicatorDescriptor(device, ctypes.byref(config))
    comms = lib.get_communicator(descriptor)
    rank = c_int(0)
    comm_size = c_int(0)
    for i in range(config.num_gpus):
        lib.get_comm_rank(descriptor, ctypes.byref(comms[i]), ctypes.byref(rank))
        lib.get_comm_size(descriptor, ctypes.byref(comms[i]), ctypes.byref(comm_size))
        print("communicator size is:", comm_size.value, "communicator rank is:", rank.value)

    data_size = 10
    send_data_gpu = []
    recv_data_gpu = []

    recv_data_host = np.empty(data_size, dtype=np.float32)

    streams = [cuda.Stream() for _ in range(config.num_gpus)]

    for i, device_id in enumerate(devices_data):
        torch.cuda.set_device(device_id)
        print(device_id)
        if device_id == 0:
            send_data_host = torch.rand(data_size, dtype=torch.float32)
            send_data_gpu = send_data_host.cuda()
            print(send_data_gpu)
        else:
            recv_data_gpu = torch.empty(data_size, dtype=torch.float32).cuda()
            print(recv_data_gpu)

    
    for i, device_id in enumerate(devices_data):
        torch.cuda.set_device(device_id)
        if device_id == 0:
            print("start send", flush = True)
            send_data_gpu_ptr = send_data_gpu.data_ptr()
            lib.Send(descriptor, send_data_gpu_ptr, data_size, datatypes.CCL_FLOAT, 1, ctypes.byref(comms[i]), None)
            print("send finished", flush = True)
        elif device_id == 1:
            print("start recv", flush = True)
            recv_data_gpu_ptr = recv_data_gpu.data_ptr()
            lib.Recv(descriptor, recv_data_gpu_ptr, data_size, datatypes.CCL_FLOAT, 0, ctypes.byref(comms[i]), None, None)
            
    torch.cuda.synchronize()

    for i, device_id in enumerate(devices_data):
        cuda.Device(device_id)
        streams[i].synchronize()
        cuda.memcpy_dtoh(recv_data_host, recv_data_gpu[i])
        print(f"rank {i} receives data: {recv_data_host.tolist()}")

    lib.destroyCommunicatorDescriptor(descriptor)

def worker():
    dl = ctypes.cdll.LoadLibrary
    lib = open_lib()
    lib.createCommunicatorDescriptor.restype = ctypes.POINTER(CommunicatorDescriptor)
    lib.communicator_init.restype = c_void_p
    lib.communicator_init.argtypes = [c_void_p]
    lib.get_communicator.restype = ctypes.POINTER(Communicator)
    lib.get_communicator.argtypes = [c_void_p]
    lib.createSendDescriptor.restype = ctypes.POINTER(SendDescriptor)
    lib.Send.restype = c_void_p
    lib.Send.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, ctypes.POINTER(Communicator), c_void_p]
    lib.createRecvDescriptor.restype = ctypes.POINTER(RecvDescriptor)
    lib.Recv.restype = c_void_p
    lib.Recv.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, ctypes.POINTER(Communicator), c_void_p, c_void_p]
    lib.get_comm_rank.restype = c_void_p
    lib.get_comm_rank.argtypes = [c_void_p, c_void_p, ctypes.POINTER(ctypes.c_int)]
    print("Start test", flush = True)
    #test_cpu(lib)    

    print("Start Gpu test", flush = True)
    test_gpu(lib)

if __name__ == "__main__":
    worker()
    