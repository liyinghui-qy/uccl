#ifndef __COMMON_CPU_H__
#define __COMMON_CPU_H__

#include <cmath>
#include <cstdint>
#include "data_type.h"
#include <mpi.h>

MPI_Datatype ccl_to_mpi_datatype(CCLDatatype datatype);

MPI_Op ccl_to_mpi_op(CCLOp op);

#endif // __COMMON_CPU_H__
