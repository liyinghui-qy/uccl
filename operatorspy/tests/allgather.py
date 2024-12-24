import ctypes
from ctypes import *
import sys
import os
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from datatypes import *
from operatorspy import (
    open_lib,
    DeviceEnum,
)

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

    lib.Allgather(descriptor, ctypes.byref(send_data), data_size, datatypes.CCL_INT, ctypes.byref(recv_data), data_size, datatypes.CCL_INT, comm, None)
    python_list = [recv_data[i] for i in range(data_size * comm_size.value)]
    print("rank", rank.value, "receives data:", python_list)
    lib.destroyCommunicatorDescriptor(descriptor)

if __name__ == "__main__":
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
    