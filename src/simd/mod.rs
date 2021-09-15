#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

#[cfg(target_arch = "x86_64")]
pub use x86::*;

#[cfg(target_arch = "aarch64")]
pub use arm::*;

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

    fn print_fx4(v: f32x4) {
        unsafe {
            let x : f32 = simd_extract(v.m, 0);
            let y : f32 = simd_extract(v.m, 1);
            let z : f32 = simd_extract(v.m, 2);
            let w : f32 = simd_extract(v.m, 3);
    
            println!("{} {} {} {}", x, y, z, w);
        }
    }

    #[test]
    fn test_equal_f32() {
        {
            let a = f32x4::splat(0.0);
            let b = f32x4::splat(0.0);
            assert!(f32x4::equal(a, b));
        }

        {
            let a = f32x4::set(1.2, 3.42, 0.0, 9.999);
            let b = f32x4::set(1.2, 3.42, 0.0, 9.999);
            assert!(f32x4::equal(a, b));
        }

        {
            let a = f32x4::set(1.2, 3.42, 0.0, 9.999);
            let b = f32x4::set(142.2, 3.42, 0.0, 42.223);
            assert!(f32x4::equal(a, b) == false);
        }
    }

    #[test]
    fn test_hmin_f32() {
        let a = f32x4::set(-4.2, 15.0, 0.0, 3.92);
        let hmin = f32x4::hmin(a);

        assert_eq!(-4.2, hmin);
    }

    // #[test]
    // fn test_vblendq_i32() {
    //     let a = i32x4::set(4, 3, 4, 5);
    //     let b = i32x4::set(8, 5, 7, 2);

    //     let mask = i32x4::cmp_lt(a , b);
    //     let blend = i32x4::blendv(a, b, mask);

    //     unsafe {
    //         let channel_0 : i32 = simd_extract(blend.m, 0);
    //         let channel_1 : i32 = simd_extract(blend.m, 1);
    //         let channel_2 : i32 = simd_extract(blend.m, 2);
    //         let channel_3 : i32 = simd_extract(blend.m, 3);
        
    //         assert_eq!(4, channel_0);
    //         assert_eq!(3, channel_1);
    //         assert_eq!(4, channel_2);
    //         assert_eq!(2, channel_3);
    //     }
    // }

    fn print_ux4(v: u32x4) {
        unsafe {
            let x : i64 = simd_extract(v.m, 0);
            let y : i64 = simd_extract(v.m, 1);
            let z : i64 = simd_extract(v.m, 2);
            let w : i64 = simd_extract(v.m, 3);
    
            println!("{} {} {} {}", x, y, z, w);
        }
    }

    #[test]
    fn test_any_u32() {
        {
            println!("Test 1");
            let a = f32x4::splat(0.0);
            let mask = f32x4::cmp_gt(a, a);
            print_ux4(mask);
            assert_eq!(mask.any(), false);
        }
        {
            println!("Test 2");
            let a = f32x4::splat(5.0);
            let b = f32x4::splat(5.0);
            let mask = f32x4::cmp_eq(a, b);
            print_ux4(mask);
            assert_eq!(mask.any(), true);
        }
        {
            println!("Test 3");
            let a = f32x4::set(5.0, 3.0, 4.0, 5.0);
            let b = f32x4::set(1.0, 2.0, 3.0, 4.0);
            let mask = f32x4::cmp_lt(a , b);
            print_ux4(mask);
            assert_eq!(mask.any(), false);
        }
        {
            println!("Test 4");
            let a = f32x4::set(5.0, 3.0, 4.0, 5.0);
            let b = f32x4::set(1.0, 2.0, 3.0, 6.0);
            let mask = f32x4::cmp_lt(a , b);
            print_ux4(mask);
            assert_eq!(mask.any(), true);
        }
    }
}