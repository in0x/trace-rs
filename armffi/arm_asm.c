#include <arm_neon.h>
#include <stdint.h>

uint32x4_t ffi_vextq2_u32(uint32x4_t a, uint32x4_t b)
{
    return vextq_u32(a, b, 2);
}

uint32x4_t ffi_vextq3_u32(uint32x4_t a, uint32x4_t b)
{
    return vextq_u32(a, b, 3);
}

float32x4_t ffi_vsqrtq_f32(float32x4_t a)
{
    return vsqrtq_f32(a);
}

float32x4_t ffi_vblendq_f32(float32x4_t a, float32x4_t b, uint32x4_t mask)
{
    uint16x8_t mask16 = vreinterpretq_u16_s32(mask);
    uint8x16_t a16 = vreinterpretq_u16_f32(a);
    uint8x16_t b16 = vreinterpretq_u16_f32(b);
    return vreinterpretq_f32_u8(vbslq_u8(mask16, a16, b16));
}

int32x4_t ffi_vblendq_i32(int32x4_t a, int32x4_t b, uint32x4_t mask)
{
    uint16x8_t mask16 = vreinterpretq_u16_s32(mask);
    uint8x16_t a16 = vreinterpretq_u16_s32(a);
    uint8x16_t b16 = vreinterpretq_u16_s32(b);
    return vreinterpretq_s32_u8(vbslq_u8(mask16, a16, b16));
}

void ffi_vst1q_s32(int32_t* ptr, int32x4_t v)
{
    vst1q_s32(ptr, v);
}

void ffi_vst1q_f32(float* ptr, float32x4_t v)
{
    vst1q_f32(ptr, v);
}

float32x4_t ffi_swizzle(float32x4_t v, uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3)
{
    static const uint32_t control_element[4] =
    {
        0x03020100,
        0x07060504,
        0x0B0A0908,
        0x0F0E0D0C,
    };

    uint8x8x2_t tbl;
    tbl.val[0] = vget_low_f32(v);
    tbl.val[1] = vget_high_f32(v);

    uint32x2_t idx = vcreate_u32( ((uint64_t)control_element[e0]) | (((uint64_t)control_element[e1]) << 32) );
    const uint8x8_t low = vtbl2_u8( tbl, idx );

    idx = vcreate_u32( ((uint64_t)control_element[e2]) | (((uint64_t)control_element[e3]) << 32) );
    const uint8x8_t high = vtbl2_u8( tbl, idx );

    return vcombine_f32(low, high);
}