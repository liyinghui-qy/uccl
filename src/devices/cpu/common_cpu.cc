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

MPI_Op ccl_to_mpi_op(CCLOp op) {
    switch (op) {
        case CCL_OP_NULL:
            return MPI_OP_NULL;
        case CCL_MAX:
            return MPI_MAX;
        case CCL_MIN:
            return MPI_MIN;
        case CCL_SUM:
            return MPI_SUM;
        case CCL_PROD:
            return MPI_PROD;
        case CCL_LAND:
            return MPI_LAND;
        case CCL_BAND:
            return MPI_BAND;
        case CCL_LOR:
            return MPI_LOR;
        case CCL_BOR:
            return MPI_BOR;
        case CCL_LXOR:
            return MPI_LXOR;
        case CCL_BXOR:
            return MPI_BXOR;
        case CCL_MINLOC:
            return MPI_MINLOC;
        case CCL_MAXLOC:
            return MPI_MAXLOC;
        case CCL_REPLACE:
            return MPI_REPLACE;
        default:
            return 0;
            break;
    }
}