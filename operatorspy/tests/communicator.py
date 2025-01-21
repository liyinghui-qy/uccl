import ctypes
from ctypes import *
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    CTensor,
    DeviceEnum,
)

class CommunicatorDescriptor(Structure):
    _fields_ = [("device", c_int)]

class Config(ctypes.Structure):
    _fields_ = [("num_gpus", ctypes.c_int),
                ("devices", ctypes.POINTER(ctypes.c_int))]
    
class Communicator(ctypes.Structure):
    _fields_ = [("deviceType", ctypes.c_int),
                ("deviceID", ctypes.c_uint),
                ("comm", ctypes.POINTER(ctypes.c_void_p))]

def test(lib, descriptor, torch_device):
    lib.communicator_init(descriptor)
    comm = lib.get_communicator(descriptor)
    a = ctypes.c_int(0)
    b = ctypes.c_int(0)
    lib.get_comm_size(descriptor, comm, ctypes.byref(a))
    lib.get_comm_rank(descriptor, comm, ctypes.byref(b))
    print("communicator size is:", a.value, "; communicator rank is:", b.value)


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    descriptor = lib.createCommunicatorDescriptor(device, None)
    test(lib, descriptor, "cpu")
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
    a = ctypes.c_int(0)
    b = ctypes.c_int(0)
    for i in range(config.num_gpus):
        lib.get_comm_size(descriptor, ctypes.byref(comms[i]), ctypes.byref(a))
        lib.get_comm_rank(descriptor, ctypes.byref(comms[i]), ctypes.byref(b))
        print("communicator size is:", a.value, "communicator rank is:", b.value)
        
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
    test_gpu(lib)

if __name__ == "__main__":
    worker()
    