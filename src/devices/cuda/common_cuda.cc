#include "common_cuda.h"


ncclDataType_t ccl_to_cuda_datatype(CCLDatatype datatype) {
    switch(datatype) {
        case CCL_CHAR:
            return ncclChar;
        case CCL_INT:
            return ncclInt;
        case CCL_HALF:
            return ncclHalf;
        case CCL_FLOAT:
            return ncclFloat;
        case CCL_DOUBLE:
            return ncclDouble;
        case CCL_INT8:
            return ncclUint8;
        case CCL_INT32:
            return ncclInt32;
        case CCL_UINT32:
            return ncclUint32;
        case CCL_INT64:
            return ncclInt64;
        case CCL_UINT64:
            return ncclUint64;
        case CCL_FLOAT16:
            return ncclFloat16;
        case CCL_BFLOAT16:
            return ncclBfloat16;
        default:
            return ncclChar;
    }
}

ncclRedOp_t ccl_to_cuda_op(CCLOp op) {
    switch(op) {
        case CCL_SUM:
            return ncclSum;
        case CCL_PROD:
            return ncclProd;
        case CCL_MAX:
            return ncclMax;
        case CCL_MIN:
            return ncclMin;
        default:
            return ncclSum;
    }
}