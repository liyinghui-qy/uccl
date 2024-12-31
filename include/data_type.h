#ifndef __CCL__DATA_TYPE_H__
#define __CCL__DATA_TYPE_H__

typedef struct DataLayout {
    unsigned short
        packed : 8,
        sign : 1,
        size : 7,
        mantissa : 8,
        exponent : 8;
} DataLayout;

// clang-format off
const static struct DataLayout
    I8   = {1, 1, 1,  7,  0},
    I16  = {1, 1, 2, 15,  0},
    I32  = {1, 1, 4, 31,  0},
    I64  = {1, 1, 8, 63,  0},
    U8   = {1, 0, 1,  8,  0},
    U16  = {1, 0, 2, 16,  0},
    U32  = {1, 0, 4, 32,  0},
    U64  = {1, 0, 8, 64,  0},
    F16  = {1, 1, 2, 10,  5},
    BF16 = {1, 1, 2,  7,  8},
    F32  = {1, 1, 4, 23,  8},
    F64  = {1, 1, 8, 52, 11};
// clang-format on

enum CCLDatatype {
    CCL_CHAR,
    CCL_SIGNED_CHAR,
    CCL_UNSIGNED_CHAR,
    CCL_BYTE,
    CCL_WCHAR,
    CCL_SHORT,
    CCL_UNSIGNED_SHORT,
    CCL_INT,
    CCL_UNSIGNED,
    CCL_LONG,
    CCL_UNSIGNED_LONG,
    CCL_FLOAT,
    CCL_DOUBLE,
    CCL_LONG_DOUBLE,
    CCL_LONG_LONG_INT,
    CCL_UNSIGNED_LONG_LONG,
    CCL_LONG_LONG,
    CCL_HALF,
    CCL_INT8,
    CCL_UINT8,
    CCL_INT32,
    CCL_UINT32,
    CCL_INT64,
    CCL_UINT64,
    CCL_FLOAT16,
    CCL_BFLOAT16
};

enum CCLOp { 
  CCL_OP_NULL,
  CCL_MAX,
  CCL_MIN,
  CCL_SUM,
  CCL_PROD,
  CCL_LAND,
  CCL_BAND,
  CCL_LOR,
  CCL_BOR,
  CCL_LXOR,
  CCL_BXOR,
  CCL_MINLOC,
  CCL_MAXLOC,
  CCL_REPLACE
};


#endif// __DATA_TYPE_H__
