import ctypes
from ctypes import *
import sys
import os
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
    send_array_type = ctypes.c_int * 16
    send_data = send_array_type(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    recv_array_type = ctypes.c_int *4
    recv_data = recv_array_type(0, 0, 0, 0)
    lib.Scatter(descriptor, ctypes.byref(send_data), 4, datatypes.CCL_INT, ctypes.byref(recv_data), 4, datatypes.CCL_INT, 0, comm, None)
    if rank.value == 0:
        python_list = [send_data[i] for i in range(16)]
        print("scattered data is:", python_list)
    python_list = [recv_data[i] for i in range(4)]
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
    lib.createScatterDescriptor.restype = c_void_p
    lib.Scatter.restype = c_void_p
    lib.Scatter.argtypes = [c_void_p, c_void_p, c_int, c_int, c_void_p, c_int, c_int, c_int, c_void_p, c_void_p]
    
    print("Start test", flush = True)
    test_cpu(lib)
    