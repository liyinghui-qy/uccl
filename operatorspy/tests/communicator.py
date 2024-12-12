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

class CommunicatorDescriptor(Structure):
    _fields_ = [("device", c_int)]

def test(lib, descriptor, torch_device):
    lib.communicator_init(descriptor)
    comm = lib.get_communicator(descriptor)
    a = ctypes.c_int(0)
    lib.get_comm_size(descriptor, comm, ctypes.byref(a))
    print("communicator size is:")
    print(a.value)
    


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    descriptor = lib.createCommunicatorDescriptor(device, None)
    test(lib, descriptor, "cpu")
    lib.destroyCommunicatorDescriptor(descriptor)


if __name__ == "__main__":
    dl = ctypes.cdll.LoadLibrary
    lib = dl("/home/liyinghui/project/operators/build/linux/x86_64/release/liboperators.so")
    print("load succeed!")
    lib.createCommunicatorDescriptor.restype = ctypes.POINTER(CommunicatorDescriptor)
    lib.communicator_init.restype = c_void_p
    lib.communicator_init.argtypes = [c_void_p]
    lib.get_communicator.restype = c_void_p
    lib.get_communicator.argtypes = [c_void_p]
    lib.get_comm_size.restype = c_void_p
    lib.get_comm_size.argtypes = [c_void_p, c_void_p, ctypes.POINTER(ctypes.c_int)]
    test_cpu(lib)
    