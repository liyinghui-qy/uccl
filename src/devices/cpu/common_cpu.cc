#include "common_cpu.h"

float f16_to_f32(uint16_t code) {
    union {
        uint32_t u32;
        float f32;
    } ans{0};
    ans.u32 = ((static_cast<uint32_t>(code) << 16) & (1 << 31)) |
              ((((code >> 10) & mask_low(5)) - 15 + 127) << 23) |
              ((code & mask_low(10)) << 13);
    return ans.f32;
}

uint16_t f32_to_f16(float val) {
    union {
        float f32;
        uint32_t u32;
    } x{val};
    return (static_cast<uint16_t>(x.u32 >> 16) & (1 << 15)) |
           (((static_cast<uint16_t>(x.u32 >> 23) - 127 + 15) & mask_low(5)) << 10) |
           (static_cast<uint16_t>(x.u32 >> 13) & mask_low(10));
}
