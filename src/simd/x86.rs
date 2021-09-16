use std::arch::x86_64::*;

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct f32x4 { pub m: __m128 }

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct i32x4 { pub m: __m128i }

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct u32x4 { pub m: __m128i }

// @TODO how does the compiler do with inlining here?

macro_rules! MY_MM_SHUFFLE {
    ($fp3: expr, $fp2: expr, $fp1: expr, $fp0: expr) => {
        (($fp3 << 6) | ($fp2 << 4) | ($fp1 << 2) | $fp0)
    };
}

impl f32x4 {
    #[inline(always)]
    pub fn splat(v: f32) -> f32x4 {
    unsafe {
        f32x4{ m: _mm_set_ps1(v) }
    }}   

    #[inline(always)]
    pub fn set(x: f32, y: f32, z: f32, w: f32) -> f32x4 {
    unsafe {
        f32x4{ m: _mm_set_ps(w, z, y, x) }
    }}

    #[inline(always)]
    pub fn loadu(ptr: &[f32]) -> f32x4 {
        assert!(ptr.len() >= 4);
    unsafe {
        f32x4 { m: _mm_loadu_ps(ptr.as_ptr()) }
    }}

    #[inline(always)]
    pub fn add(a: f32x4, b: f32x4) -> f32x4 {
    unsafe {
        f32x4 { m: _mm_add_ps(a.m, b.m) }
    }}

    #[inline(always)]
    pub fn sub(a: f32x4, b: f32x4) -> f32x4 {
    unsafe {
        f32x4 { m: _mm_sub_ps(a.m, b.m) }
    }}

    #[inline(always)]
    pub fn mul(a: f32x4, b: f32x4) -> f32x4 {
    unsafe {
        f32x4 { m: _mm_mul_ps(a.m, b.m) }
    }}

    #[inline(always)]
    pub fn div(a: f32x4, b: f32x4) -> f32x4 {
    unsafe {
        f32x4 { m: _mm_div_ps(a.m, b.m) }
    }}
    
    #[inline(always)]
    pub fn cmp_gt(a: f32x4, b: f32x4) -> u32x4 {
    unsafe {
        u32x4 { m: _mm_castps_si128(
            _mm_cmpgt_ps(a.m, b.m)) }
    }}

    #[inline(always)]
    pub fn cmp_lt(a: f32x4, b: f32x4) -> u32x4 {
    unsafe {
        u32x4 { m: _mm_castps_si128(
            _mm_cmplt_ps(a.m, b.m)) }
    }}

    #[inline(always)]
    pub fn cmp_le(a: f32x4, b: f32x4) -> u32x4 {
    unsafe {
        u32x4 { m: _mm_castps_si128(
            _mm_cmple_ps(a.m, b.m)) }
    }}
    
    #[inline(always)]
    pub fn cmp_eq(a: f32x4, b: f32x4) -> u32x4 {
    unsafe {
        u32x4 { m: _mm_castps_si128(
            _mm_cmpeq_ps(a.m, b.m)) }
    }}
   
    #[inline(always)]
    pub fn sqrt(a: f32x4) -> f32x4 {
    unsafe {
        f32x4 {
            m: _mm_sqrt_ps(a.m)
        }
    }}
    
    #[inline(always)]
    pub fn blendv(a: f32x4, b: f32x4, mask: u32x4) -> f32x4 {
    unsafe {
        f32x4 {
            m: _mm_blendv_ps(b.m, a.m, _mm_castsi128_ps(mask.m))
        }
    }}

    // @Cleanup _mm_shuffle_x takes a constant as its mask, which rust
    // does through its generics. So if we want swizzle we need to reimplement it
    // the way we did under arm. But it doesnt even seem like we use swizzle anymore.
    // pub fn swizzle(v: f32x4, c0: u32, c1: u32, c2: u32, c3: u32) -> f32x4 {
    //     assert!(c0 < 4); assert!(c1 < 4);
    //     assert!(c2 < 4); assert!(c3 < 4);
    // unsafe {
    //     f32x4 { m: _mm_castsi128_ps(
    //         _mm_shuffle_epi32(
    //             _mm_castps_si128(v.m),
    //             MY_MM_SHUFFLE!(c3, c2, c1, c0))) }
    // }}
    
    #[inline(always)]
    pub fn max(a: f32x4, b: f32x4) -> f32x4 {
    unsafe {
        f32x4 { m: _mm_max_ps(a.m, b.m) }
    }}
    
    #[inline(always)]
    pub fn min(a: f32x4, b: f32x4) -> f32x4 {
    unsafe {
        f32x4 { m: _mm_min_ps(a.m, b.m) }
    }}

    /// Returns a scalar containing the smallest channel in the lane.  
    pub fn hmin(v: f32x4) -> f32 {
    unsafe {
        let mut m : __m128 = v.m;
        
        const MASK1: i32 = MY_MM_SHUFFLE!(2, 1, 0, 3);
        m = _mm_min_ps(m, _mm_shuffle_ps::<MASK1>(m, m));
        
        const MASK2: i32 = MY_MM_SHUFFLE!(1, 0, 3, 2);
        m = _mm_min_ps(m, _mm_shuffle_ps::<MASK2>(m, m));
        
        _mm_cvtss_f32(m)
    }}
    
    // unsafe fn hmax(mut m: __m128) -> f32 {
    //     m = _mm_max_ps(m, MY_SHUFFLE_4!(m, 2, 3, 0, 0));
    //     m = _mm_max_ps(m, MY_SHUFFLE_4!(m, 1, 0, 0, 0));
    //     _mm_cvtss_f32(m)
    // }

    #[inline(always)]
    pub fn equal(a: f32x4, b: f32x4) -> bool {
    unsafe {
        let eq_mask = _mm_cmpeq_ps(a.m, b.m);
        _mm_movemask_epi8(_mm_castps_si128(eq_mask)) == 0xffff
    }}

    #[inline(always)]
    pub fn storeu(v: f32x4, mem: &mut [f32; 4]) {
    unsafe {
        _mm_storeu_ps(mem.as_mut_ptr(), v.m)
    }}
}

impl i32x4 {
    #[inline(always)]
    pub fn splat(i: i32) -> i32x4 {
    unsafe {
        i32x4 { m: _mm_set1_epi32(i) }
    }}

    #[inline(always)]
    pub fn set(x: i32, y: i32, z: i32, w: i32) -> i32x4 {
    unsafe {
        i32x4 { m: _mm_set_epi32(w, z, y, x) }
    }}

    #[inline(always)]
    pub fn add(a: i32x4, b: i32x4) -> i32x4 {
    unsafe {
        i32x4 { m: _mm_add_epi32(a.m, b.m) }
    }}

    #[inline(always)]
    pub fn cmp_lt(a: i32x4, b: i32x4) -> u32x4 {
    unsafe {
        u32x4 { m: _mm_cmplt_epi32(a.m, b.m) }
    }}

    #[inline(always)]
    pub fn blendv(a: i32x4, b: i32x4, mask: u32x4) -> i32x4 {
    unsafe {
        i32x4 { m: _mm_blendv_epi8(a.m, b.m, mask.m) }
    }}

    #[inline(always)]
    pub fn storeu(v: i32x4, mem: &mut [i32; 4]) {
        unsafe {
            _mm_storeu_epi32(mem.as_mut_ptr(), v.m);
        }
    }
}

impl u32x4 {
    #[inline(always)]
    pub fn set(x: u32, y: u32, z: u32, w: u32) -> u32x4 {
    unsafe {
      u32x4 { m: _mm_set_epi32(w as i32, z as i32, y as i32, x as i32) }  
    }}

    #[inline(always)]
    pub fn any(&self) -> bool {
    unsafe {
        _mm_movemask_epi8(self.m) != 0
    }}

    #[inline(always)]
    pub fn all(&self) -> bool {
    unsafe {
        _mm_movemask_epi8(self.m) == 15
    }}

    #[inline(always)]
    pub fn movemask(flags: u32x4) -> u32 {
    unsafe {
        _mm_movemask_epi8(flags.m) as u32
    }
    }

    #[inline(always)]
    pub fn and(a: u32x4, b: u32x4) -> u32x4 {
    unsafe {
        u32x4 { m: _mm_and_si128(a.m, b.m) }
    }}

    #[inline(always)]
    pub fn or(a: u32x4, b: u32x4) -> u32x4 {
    unsafe {
        u32x4 { m: _mm_or_si128(a.m, b.m) }
    }}

    #[inline(always)]
    pub fn not(a: u32x4) -> u32x4 {
    unsafe {
        u32x4 { m: _mm_xor_si128(a.m, _mm_set1_epi32(-1)) }
    }}

    #[inline(always)]
    pub fn get_flags(a: u32x4) -> [bool;4] {
    unsafe {
        let mut store : [i32;4] = [0, 0, 0 ,0];
        _mm_storeu_epi32(store.as_mut_ptr(), a.m);

        [store[0] != 0, store[1] != 0, store[2] != 0, store[3] != 0]
    }}
}
