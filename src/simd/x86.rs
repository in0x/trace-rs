use std::arch::x86_64::*;

#[derive(Copy, Clone)]
pub struct f32x4(__m128);

#[derive(Copy, Clone)]
pub struct i32x4(__m128i);

impl f32x4 {
    pub fn splat(v: f32) -> f32x4 {
        f32x4(_mm_set_ps1(v))
    }   
}

macro_rules! MY_MM_SHUFFLE {
    ($fp3: expr, $fp2: expr, $fp1: expr, $fp0: expr) => {
        (($fp3 << 6) | ($fp2 << 4) | ($fp1 << 2) | $fp0)
    };
}

macro_rules! MY_SHUFFLE_4 {
    ($v: expr, $x: expr, $y: expr, $z: expr, $w: expr) => {
        _mm_shuffle_ps($v, $v, MY_MM_SHUFFLE!($w,$z,$y,$x))
    };
}

fn select(a: __m128, b: __m128, cond: __m128) -> __m128 {
    unsafe {
        _mm_blendv_ps(a, b, cond)
    }
}

fn selecti(a: __m128i, b: __m128i, cond: __m128) -> __m128i {
    unsafe {
        _mm_blendv_epi8(a, b, _mm_castps_si128(cond))
    }
}

fn hmin(mut m: __m128) -> f32 {
    unsafe {
        m = _mm_min_ps(m, MY_SHUFFLE_4!(m, 2, 3, 0, 0));
        m = _mm_min_ps(m, MY_SHUFFLE_4!(m, 1, 0, 0, 0));
        _mm_cvtss_f32(m)
    }
}