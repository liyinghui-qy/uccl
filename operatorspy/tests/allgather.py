import ctypes
from ctypes import *
import sys
import os
import random
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
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
    devices_type = c_int * config.num_gpus
    config.devices = devices_type(*devices_data)
    descriptor = lib.createCommunicatorDescriptor(device, ctypes.byref(config))

    comm = lib.get_communicator(descriptor)
    rank = c_int(0)
    lib.get_comm_rank(descriptor, comm, ctypes.byref(rank))
    comm_size = c_int(0)
    lib.get_comm_size(descriptor, comm, ctypes.byref(comm_size))

    data_size = 4
    send_data_gpu = [None] * config.num_gpus
    recv_data_gpu = [None] * config.num_gpus

    recv_data_host = np.empty(data_size * comm_size.value, dtype=np.float32)

    streams = [cuda.Stream() for _ in range(config.num_gpus)]

    for i, device_id in enumerate(devices_data):
        send_data_host = np.random.rand(data_size).astype(np.float32)
        cuda.Device(device_id)
        send_data_gpu[i] = cuda.mem_alloc(data_size * np.float32().nbytes)
        recv_data_gpu[i] = cuda.mem_alloc(data_size * comm_size.value * np.float32().nbytes)
        cuda.memcpy_htod(send_data_gpu[i], send_data_host)
        print(f"rank {i} sends data: {send_data_host.tolist()}")

    for i, device_id in enumerate(devices_data):
        cuda.Device(device_id)
        send_data_gpu_ptr = c_void_p(int(send_data_gpu[i]))
        recv_data_gpu_ptr = c_void_p(int(recv_data_gpu[i]))

        lib.Allgather(
            descriptor,
            send_data_gpu_ptr,
            data_size,
            datatypes.CCL_FLOAT,
            recv_data_gpu_ptr,
            data_size * comm_size.value,
            datatypes.CCL_FLOAT,
            comm[i],
            streams[i].handle
        )

    for i, device_id in enumerate(devices_data):
        cuda.Device(device_id)
        streams[i].synchronize()
        cuda.memcpy_dtoh(recv_data_host, recv_data_gpu[i])
        print(f"rank {i} receives data: {recv_data_host.tolist()}")

    for i, device_id in enumerate(devices_data):
        cuda.Device(device_id)

        if send_data_gpu[i] is not None:
            send_data_gpu[i].free()
        
        if recv_data_gpu[i] is not None:
            recv_data_gpu[i].free()

    lib.destroyCommunicatorDescriptor(descriptor)

def worker():
    dl = ctypes.cdll.LoadLibrary
    lib = open_lib()
    lib.createCommunicatorDescriptor.restype = ctypes.POINTER(CommunicatorDescriptor)
    lib.createCommunicatorDescriptor.argtypes = [ctypes.c_int, c_void_p]
    lib.communicator_init.restype = c_void_p
    lib.communicator_init.argtypes = [c_void_p]
    lib.get_communicator.restype = ctypes.POINTER(Communicator)
    lib.get_communicator.argtypes = [c_void_p]
    lib.get_comm_size.restype = c_void_p
    lib.get_comm_size.argtypes = [c_void_p, c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.get_comm_rank.restype = c_void_p
    lib.get_comm_rank.argtypes = [c_void_p, c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.Allgather.restype = c_void_p
    lib.Allgather.argtypes = [c_void_p, c_void_p, c_int, c_int, c_void_p, c_int, c_int, ctypes.POINTER(Communicator), c_void_p]
    test_gpu(lib)

if __name__ == "__main__":
    print("Start test cpu")
    dl = ctypes.cdll.LoadLibrary
    lib = open_lib()
    lib.createCommunicatorDescriptor.restype = ctypes.POINTER(CommunicatorDescriptor)
    lib.communicator_init.restype = c_void_p
    lib.communicator_init.argtypes = [c_void_p]
    lib.get_communicator.restype = c_void_p
    lib.get_communicator.argtypes = [c_void_p]
    lib.get_comm_rank.restype = c_void_p
    lib.get_comm_rank.argtypes = [c_void_p, c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.get_comm_size.restype = c_void_p
    lib.get_comm_size.argtypes = [c_void_p, c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.createAlltoallDescriptor.restype = c_void_p
    lib.Allgather.restype = c_void_p
    lib.Allgather.argtypes = [c_void_p, c_void_p, c_int, c_int, c_void_p, c_int, c_int, c_void_p, c_void_p]
    print("Start test", flush = True)
    test_cpu(lib)

    print("Start test gpu")
    worker()