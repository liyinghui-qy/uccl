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

class SendDescriptor(Structure):
    _fields_ = [("device", c_int)]

class RecvDescriptor(Structure):
    _fields_ = [("device", c_int)]

def test_send(lib, descriptor, torch_device):
    comm = lib.get_communicator(descriptor)
    a = ctypes.c_int(100)
    lib.Send(descriptor, ctypes.byref(a), 1, 0x4c000405, 1, comm)
    print("Send data", a.value, "to rank 1", flush = True)

def test_recv(lib, descriptor, torch_device):
    comm = lib.get_communicator(descriptor)
    b = ctypes.c_int(0)
    lib.Recv(descriptor, ctypes.byref(b), 1, 0x4c000405, 0, comm, None)
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

def worker():
    dl = ctypes.cdll.LoadLibrary
    lib = open_lib()
    lib.createCommunicatorDescriptor.restype = ctypes.POINTER(CommunicatorDescriptor)
    lib.communicator_init.restype = c_void_p
    lib.communicator_init.argtypes = [c_void_p]
    lib.get_communicator.restype = c_void_p
    lib.get_communicator.argtypes = [c_void_p]
    lib.createSendDescriptor.restype = ctypes.POINTER(SendDescriptor)
    lib.Send.restype = c_void_p
    lib.Send.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_void_p]
    lib.createRecvDescriptor.restype = ctypes.POINTER(RecvDescriptor)
    lib.Recv.restype = c_void_p
    lib.Recv.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_void_p, c_void_p]
    lib.get_comm_rank.restype = c_void_p
    lib.get_comm_rank.argtypes = [c_void_p, c_void_p, ctypes.POINTER(ctypes.c_int)]
    print("Start test", flush = True)
    test_cpu(lib)

if __name__ == "__main__":
    worker()
    