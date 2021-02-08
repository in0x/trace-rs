use std::arch::aarch64::*;

extern "platform-intrinsic" {
    pub fn simd_extract<T, U>(x: T, idx: u32) -> U;
    pub fn simd_div<T>(x: T, y: T) -> T;
}

#[allow(improper_ctypes)]
#[link(name = "armffi")]
extern "C" {
    fn ffi_vextq2_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    fn ffi_vextq3_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    fn ffi_vsqrtq_f32(a: float32x4_t) -> float32x4_t; 
    fn ffi_vblendq_f32(a: float32x4_t, b: float32x4_t, mask: uint32x4_t) -> float32x4_t;
    fn ffi_vblendq_i32(a: int32x4_t, b: int32x4_t, mask: uint32x4_t) -> int32x4_t;
    fn ffi_swizzle(v: float32x4_t, e0: u32, e1: u32, e2: u32, e3: u32) -> float32x4_t;
    fn ffi_vst1q_s32(ptr: * mut i32, v: int32x4_t);
    fn ffi_vst1q_f32(ptr: * mut f32, v: float32x4_t);
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct f32x4 { pub m: float32x4_t }

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct i32x4 { pub m: int32x4_t }

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct u32x4 { pub m: uint32x4_t }

// macro_rules! SHUFFLE {
//     ($fp3: expr, $fp2: expr, $fp1: expr, $fp0: expr) => {
//         (($fp3 << 6) | ($fp2 << 4) | ($fp1 << 2) | $fp0)
//     };
// }

// macro_rules! SHUFFLE_4 {
//     ($v: expr, $x: expr, $y: expr, $z: expr, $w: expr) => {
//         _mm_shuffle_ps($v, $v, MY_MM_SHUFFLE!($w,$z,$y,$x))
//     };
// }

impl f32x4 {
    pub fn splat(v: f32) -> f32x4 {
        unsafe {
            f32x4{m: std::mem::transmute([v, v, v, v])}
        }
    }   

    pub fn set(x: f32, y: f32, z: f32, w: f32) -> f32x4 {
        unsafe {
            f32x4{ m: std::mem::transmute([x, y, z, w])}
        }
    }

    pub fn loadu(ptr: &[f32]) -> f32x4 {
        assert!(ptr.len() >= 4);
        unsafe {
            f32x4 {
                m: vld1q_f32(ptr.as_ptr())
            }
        }
    }

    pub fn add(a: f32x4, b: f32x4) -> f32x4 {
        unsafe {
            f32x4 {
                m: vaddq_f32(a.m, b.m)
            }
        }
    }

    pub fn sub(a: f32x4, b: f32x4) -> f32x4 {
        unsafe {
            f32x4 {
                m: vsubq_f32(a.m, b.m)
            }
        }
    }

    pub fn mul(a: f32x4, b: f32x4) -> f32x4 {
        unsafe {
            f32x4 {
                m: vmulq_f32(a.m, b.m)
            }
        }
    }

    pub fn div(a: f32x4, b: f32x4) -> f32x4 {
        unsafe {
            f32x4 {
                m: simd_div(a.m, b.m)
            }
        }
    }

    pub fn cmp_gt(a: f32x4, b: f32x4) -> u32x4 {
        unsafe {
            u32x4 {
                m: vcgtq_f32(a.m, b.m)
            }
        }
    }

    pub fn cmp_lt(a: f32x4, b: f32x4) -> u32x4 {
        unsafe {
            u32x4 {
                m: vcltq_f32(a.m, b.m)
            }
        }
    }

    pub fn cmp_eq(a: f32x4, b: f32x4) -> u32x4 {
        unsafe {
            u32x4 {
                m: vceqq_f32(a.m, b.m)
            }
        }
    }

    pub fn sqrt(a: f32x4) -> f32x4 {
        unsafe {
            f32x4 {
                m: ffi_vsqrtq_f32(a.m)
            }
        }
    }

    pub fn blendv(a: f32x4, b: f32x4, mask: u32x4) -> f32x4 {
        unsafe {
            f32x4 {
                m: ffi_vblendq_f32(a.m, b.m, mask.m)
            }
        }
    }

    pub fn swizzle(v: f32x4, c0: u32, c1: u32, c2: u32, c3: u32) -> f32x4 {
        assert!(c0 < 4); assert!(c1 < 4);
        assert!(c2 < 4); assert!(c3 < 4);
        
        unsafe {
            f32x4 {
                m: ffi_swizzle(v.m, c0, c1, c2, c3)
            }
        }
    }

    /// Returns a scalar containing the smallest channel in the vector.  
    pub fn hmin(v: f32x4) -> f32 {
        unsafe {
            let m1 = vminq_f32(v.m, ffi_swizzle(v.m, 2, 3, 0, 0));
            let m2 = vminq_f32(m1, ffi_swizzle(m1, 1, 0, 0, 0));
            simd_extract(m2, 0)
        }
    }

    pub fn equal(a: f32x4, b: f32x4) -> bool {
        unsafe {
            let eq_mask = vceqq_f32(a.m, b.m);
            u32x4::vmovemaskq_u32(eq_mask) != 15
        }
    }

    pub fn storeu(v: f32x4, mem: &mut [f32; 4]) {
        unsafe {
            ffi_vst1q_f32(mem.as_mut_ptr(), v.m);
        }
    }
}

impl i32x4 {
    pub fn splat(i: i32) -> i32x4 {
        unsafe {
            i32x4 {
                m: std::mem::transmute([i, i, i ,i])
            }
        }
    }
    
    pub fn set(x: i32, y: i32, z: i32, w: i32) -> i32x4 {
        unsafe {
            i32x4 {
                m: std::mem::transmute([x, y, z, w])
            }
        }
    }

    pub fn add(a: i32x4, b: i32x4) -> i32x4 {
        unsafe {
            i32x4 {
                m: vaddq_s32(a.m, b.m)
            }
        }
    }

    pub fn cmp_lt(a: i32x4, b: i32x4) -> u32x4 {
        unsafe {
            u32x4 {
                m: vcltq_s32(a.m, b.m)
            }
        }
    }

    pub fn blendv(a: i32x4, b: i32x4, mask: u32x4) -> i32x4 {
        unsafe {
            i32x4 {
                m: ffi_vblendq_i32(a.m, b.m, mask.m)
            }
        }
    }

    pub fn storeu(v: i32x4, mem: &mut [i32; 4]) {
        unsafe {
            ffi_vst1q_s32(mem.as_mut_ptr(), v.m);
        }
    }
}

impl u32x4 {
    pub fn set(x: u32, y: u32, z: u32, w: u32) -> u32x4 {
        unsafe {
            u32x4 {
                m: std::mem::transmute([x, y, z, w])
            }
        }
    }

    pub fn vmovemaskq_u32(flags: uint32x4_t) -> u32 {
        unsafe {
            // We want to make a mask that has the four first bits set
            // based on whether the corresponding channel in flags was set.
            let mask_bits = u32x4::set(1, 2, 4, 8);  
            let msk = vandq_u32(flags, mask_bits.m);

            let shf1 = ffi_vextq2_u32(msk, msk); // [2 3 0 1]
            let or1  = vorrq_u32(msk, shf1);     // [0+2 1+3 0+2 1+3]
            let shf2 = ffi_vextq3_u32(or1, or1); // [1+3 0+2 1+3 0+2]
            let or2  = vorrq_u32(or1, shf2);     // [0+1+2+3 ...]
            // return vgetq_lane_u32(or2, 0);
            simd_extract(or2, 0)
        }
    }

    pub fn any(&self) -> bool {
        u32x4::vmovemaskq_u32(self.m) != 0
    }

    pub fn all(&self) -> bool {
        u32x4::vmovemaskq_u32(self.m) == 15
    }

    pub fn and(a: u32x4, b: u32x4) -> u32x4 {
        unsafe {
            u32x4 {
                m: vandq_u32(a.m, b.m)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_set_f32() {
        let v = f32x4::set(15.0, 42.0, -36.0, 1.0);
        unsafe {
            let channel_0 : f32 = simd_extract(v.m, 0);
            let channel_1 : f32 = simd_extract(v.m, 1);
            let channel_2 : f32 = simd_extract(v.m, 2);
            let channel_3 : f32 = simd_extract(v.m, 3);
        
            assert_eq!(15.0, channel_0);
            assert_eq!(42.0, channel_1);
            assert_eq!(-36.0, channel_2);
            assert_eq!(1.0, channel_3);
        }
    }

    #[test]
    fn test_splat_f32() {
        let v = f32x4::splat(42.0);
        unsafe {
            let channel_0 : f32 = simd_extract(v.m, 0);
            let channel_1 : f32 = simd_extract(v.m, 1);
            let channel_2 : f32 = simd_extract(v.m, 2);
            let channel_3 : f32 = simd_extract(v.m, 3);
        
            assert_eq!(42.0, channel_0);
            assert_eq!(42.0, channel_1);
            assert_eq!(42.0, channel_2);
            assert_eq!(42.0, channel_3);
        }
    }

    #[test]
    fn test_vblendq_f32() {
        let a = f32x4::set(5.0, 3.0, 4.0, 5.0);
        let b = f32x4::set(1.0, 2.0, 3.0, 6.0);

        let mask = f32x4::cmp_lt(a , b);
        let blend = f32x4::blendv(a, b, mask);

        unsafe {
            let channel_0 : f32 = simd_extract(blend.m, 0);
            let channel_1 : f32 = simd_extract(blend.m, 1);
            let channel_2 : f32 = simd_extract(blend.m, 2);
            let channel_3 : f32 = simd_extract(blend.m, 3);
        
            assert_eq!(1.0, channel_0);
            assert_eq!(2.0, channel_1);
            assert_eq!(3.0, channel_2);
            assert_eq!(5.0, channel_3);
        }
    }

    #[test]
    fn test_swizzle_f32() {
        let a = f32x4::set(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::set(2.0, 3.0, 1.0, 4.0);

        let c = f32x4::swizzle(a, 1, 2, 0, 3);
    
        assert!(f32x4::equal(c, b));
    }

    #[test]
    fn test_hmin_f32() {
        let a = f32x4::set(-4.2, 15.0, 0.0, 3.92);
        let hmin = f32x4::hmin(a);

        assert_eq!(-4.2, hmin);
    }

    #[test]
    fn test_vblendq_i32() {
        let a = i32x4::set(4, 3, 4, 5);
        let b = i32x4::set(8, 5, 7, 2);

        let mask = i32x4::cmp_lt(a , b);
        let blend = i32x4::blendv(a, b, mask);

        unsafe {
            let channel_0 : i32 = simd_extract(blend.m, 0);
            let channel_1 : i32 = simd_extract(blend.m, 1);
            let channel_2 : i32 = simd_extract(blend.m, 2);
            let channel_3 : i32 = simd_extract(blend.m, 3);
        
            assert_eq!(4, channel_0);
            assert_eq!(3, channel_1);
            assert_eq!(4, channel_2);
            assert_eq!(2, channel_3);
        }
    }

    fn print_x4(v: uint32x4_t) {
        unsafe {
            let x : u32 = simd_extract(v, 0);
            let y : u32 = simd_extract(v, 1);
            let z : u32 = simd_extract(v, 2);
            let w : u32 = simd_extract(v, 3);
    
            println!("{} {} {} {}", x, y, z, w);
        }
    }

    #[test]
    fn test_any_u32() {
        {
            println!("Test 1");
            let a = f32x4::splat(0.0);
            let mask = f32x4::cmp_gt(a, a);
            print_x4(mask.m);
            assert_eq!(mask.any(), false);
        }
        {
            println!("Test 2");
            let a = f32x4::splat(5.0);
            let b = f32x4::splat(5.0);
            let mask = f32x4::cmp_eq(a, b);
            print_x4(mask.m);
            assert_eq!(mask.any(), true);
        }
        {
            println!("Test 3");
            let a = f32x4::set(5.0, 3.0, 4.0, 5.0);
            let b = f32x4::set(1.0, 2.0, 3.0, 4.0);
            let mask = f32x4::cmp_lt(a , b);
            print_x4(mask.m);
            assert_eq!(mask.any(), false);
        }
        {
            println!("Test 4");
            let a = f32x4::set(5.0, 3.0, 4.0, 5.0);
            let b = f32x4::set(1.0, 2.0, 3.0, 6.0);
            let mask = f32x4::cmp_lt(a , b);
            print_x4(mask.m);
            assert_eq!(mask.any(), true);
        }
    }
}