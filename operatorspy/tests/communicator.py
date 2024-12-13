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

import torch
from multiprocessing import Process

class CommunicatorDescriptor(Structure):
    _fields_ = [("device", c_int)]

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

def worker():
    dl = ctypes.cdll.LoadLibrary
    lib = open_lib()
    lib.createCommunicatorDescriptor.restype = ctypes.POINTER(CommunicatorDescriptor)
    lib.communicator_init.restype = c_void_p
    lib.communicator_init.argtypes = [c_void_p]
    lib.get_communicator.restype = c_void_p
    lib.get_communicator.argtypes = [c_void_p]
    lib.get_comm_size.restype = c_void_p
    lib.get_comm_size.argtypes = [c_void_p, c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.get_comm_rank.restype = c_void_p
    lib.get_comm_rank.argtypes = [c_void_p, c_void_p, ctypes.POINTER(ctypes.c_int)]
    test_cpu(lib)

if __name__ == "__main__":
    worker()
    