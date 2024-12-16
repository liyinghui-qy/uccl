#include "common_cpu.h"

MPI_Datatype ccl_to_mpi_datatype(CCLDatatype datatype) {
    switch (datatype) {
        case CCL_CHAR:
            return MPI_CHAR;
        case CCL_SIGNED_CHAR:
            return MPI_SIGNED_CHAR;
        case CCL_UNSIGNED_CHAR:
            return MPI_UNSIGNED_CHAR;
        case CCL_BYTE:
            return MPI_BYTE;
        case CCL_WCHAR:
            return MPI_WCHAR;
        case CCL_SHORT:
            return MPI_SHORT;
        case CCL_UNSIGNED_SHORT:
            return MPI_UNSIGNED_SHORT;
        case CCL_INT:
            return MPI_INT;
        case CCL_UNSIGNED:
            return MPI_UNSIGNED;
        case CCL_LONG:
            return MPI_LONG;
        case CCL_UNSIGNED_LONG:
            return MPI_UNSIGNED_LONG;
        case CCL_FLOAT:
            return MPI_FLOAT;
        case CCL_DOUBLE:
            return MPI_DOUBLE;
        case CCL_LONG_DOUBLE:
            return MPI_LONG_DOUBLE;
        case CCL_LONG_LONG_INT:
            return MPI_LONG_LONG_INT;
        case CCL_UNSIGNED_LONG_LONG:
            return MPI_UNSIGNED_LONG_LONG;
        case CCL_LONG_LONG:
            return MPI_LONG_LONG;
        default:
            return 0;
            break;
    }
}