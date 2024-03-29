//! math.rs: Provides functionality for solving linear algebra problems. 

use std::cell::RefCell;
use rand::{Rng,SeedableRng};
use rand::rngs::StdRng;

fn _seedable_rand() -> f32 {
    thread_local! {
        pub static THREAD_FAST_RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(801289018191233));
    }

    let mut_ptr = THREAD_FAST_RNG.with(|r| r.as_ptr());
    unsafe { (*mut_ptr).gen() }
}

pub fn rand_f32() -> f32 {
    _seedable_rand()
}

pub fn rand_int() -> u32 {
    _seedable_rand() as u32
}

/// Finds the next largest power of base that is less than or equal to num.
pub fn next_power_of_n_le(num: usize, base: usize) -> usize {
    let mut pow = 1;

    while base.pow(pow + 1) <= num {
        pow += 1;
    }

    base.pow(pow)
}

pub fn clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min { return min; }
    if x > max { return max; }
    x
}

pub fn rad_to_degree(rad: f32) -> f32 {
    rad * (180.0 / std::f32::consts::PI)
}

pub fn degree_to_rad(degrees: f32) -> f32 {
    degrees * (std::f32::consts::PI / 180.0)
}

pub fn rand_in_range(min: f32, max: f32) -> f32 {
    let n_0_to_1 : f32 = rand_f32();
    min + n_0_to_1 * (max - min)
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Vec3 {
    pub x : f32,
    pub y : f32,
    pub z : f32
}

impl Vec3 {
    pub fn zero() -> Vec3 {
        Vec3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn one() -> Vec3 {
        Vec3 { x: 1.0, y: 1.0, z: 1.0 }
    }

    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }
    
    pub fn is_near_zero(&self) -> bool {
        let eps = 1.0e-8;
        (self.x.abs() < eps) && (self.y.abs() < eps) && (self.z.abs() < eps) 
    }

    pub fn random() -> Vec3 {
        Vec3 { x: rand_f32(), y: rand_f32(), z: rand_f32() }
    }

    pub fn random_range(min: f32, max: f32) -> Vec3 {
        Vec3 { x: rand_in_range(min, max), y: rand_in_range(min, max), z: rand_in_range(min, max) }
    }

    pub fn at(&self, idx: usize) -> f32 {
        match idx {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => unreachable!()
        }
    }
}

#[macro_export]
macro_rules! vec3 {
    ($x: expr, $y: expr, $z: expr ) => {
        Vec3 { x: $x, y: $y, z: $z}
    };

    ($replicate: expr) => {
        Vec3 {x: $replicate, y: $replicate, z: $replicate}
    };
}

macro_rules! element_wise_binary_op {
    ($op: tt, $trait: ident, $func: ident) => {
        impl std::ops::$trait<Vec3> for Vec3 {
            type Output = Vec3;

            fn $func(self, rhs: Vec3) -> Vec3 {
                Vec3 {
                    x: self.x $op rhs.x,
                    y: self.y $op rhs.y,
                    z: self.z $op rhs.z,
                }
            }
        }

        impl std::ops::$trait<f32> for Vec3 {
            type Output = Vec3;
    
            fn $func(self, rhs: f32) -> Vec3 {
                Vec3 {
                    x: self.x $op rhs,
                    y: self.y $op rhs,
                    z: self.z $op rhs,
                }
            }
        }

        impl std::ops::$trait<Vec3> for f32 {
            type Output = Vec3;
    
            fn $func(self, rhs: Vec3) -> Vec3 {
                Vec3 {
                    x: self $op rhs.x,
                    y: self $op rhs.y,
                    z: self $op rhs.z,
                }
            }
        }
    };
    // TODO(phil): Implement taking references 
}

element_wise_binary_op!(+, Add, add);
element_wise_binary_op!(-, Sub, sub);
element_wise_binary_op!(*, Mul, mul);
element_wise_binary_op!(/, Div, div);

macro_rules! element_wise_binary_op_assign {
    ($op: tt, $trait: ident, $func: ident) => {
        impl std::ops::$trait<Vec3> for Vec3 {
            fn $func(&mut self, rhs: Vec3) {
                self.x $op rhs.x;
                self.y $op rhs.y;
                self.z $op rhs.z;
            }
        }

        impl  std::ops::$trait<f32> for Vec3 {
            fn $func(&mut self, rhs: f32) {
                self.x $op rhs;
                self.y $op rhs;
                self.z $op rhs;
            }
        }
    };
    // TODO(phil): Implement taking references 
}

element_wise_binary_op_assign!(+=, AddAssign, add_assign);
element_wise_binary_op_assign!(-=, SubAssign, sub_assign);
element_wise_binary_op_assign!(*=, MulAssign, mul_assign);
element_wise_binary_op_assign!(/=, DivAssign, div_assign);

impl std::ops::Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        Vec3 { x: -self.x, y: -self.y, z: -self.z }
    }
}

pub fn length(v: Vec3) -> f32 {
    (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
}

pub fn length_sq(v: Vec3) -> f32 {
    v.x * v.x + v.y * v.y + v.z * v.z
}

pub fn normalized(v: Vec3) -> Vec3 {
    let inv_len = 1.0 / length(v);
    Vec3 {x: v.x * inv_len, y: v.y * inv_len, z: v.z * inv_len}
}

pub fn dot(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: (a.y * b.z - a.z * b.y),
        y: -(a.x * b.z - a.z * b.x),
        z: (a.x * b.y - a.y * b.x)
    }
}

pub fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - 2.0 * dot(v,n) * n
}

pub fn refract(v: Vec3, n: Vec3, ni_over_nt: f32) -> Vec3 {
    let cos_theta = f32::min(dot(-v, n), 1.0);
    let r_out_perp = ni_over_nt * (v + cos_theta * n);
    let r_out_para = -((1.0 - length_sq(r_out_perp)).abs().sqrt()) * n;
    r_out_perp + r_out_para
}

/// Returns a random point within a unit-radius sphere.
// pub fn rand_in_unit_sphere() -> Vec3 {
//     loop {
//         let p = vec3![rand_in_range(-1.0, 1.0), rand_in_range(-1.0, 1.0), rand_in_range(-1.0, 1.0)];
//         if length_sq(p) < 1.0 { return p };
//     }
// }

// http://corysimon.github.io/articles/uniformdistn-on-sphere/
pub fn rand_in_unit_sphere() -> Vec3 {
    let theta = std::f32::consts::TAU * rand_in_range(0.0, 1.0);
    let phi = f32::acos(1.0 - 2.0 * rand_in_range(0.0, 1.0));

    vec3![f32::sin(phi) * f32::cos(theta),
          f32::sin(phi) * f32::sin(theta),
          f32::cos(phi)]
}

pub fn rand_unit_vector() -> Vec3 {
    normalized(rand_in_unit_sphere())
}

pub fn rand_in_unit_disk() -> Vec3 {
    loop {
        let p = vec3![rand_in_range(-1.0, 1.0), rand_in_range(-1.0, 1.0), 0.0];
        if length_sq(p) < 1.0 { return p };
    }
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! assert_nearly_eq {
        ($v1: expr, $v2: expr, $epsilon: expr) => {
            match (&$v1, &$v2, $epsilon) {
                (lhs, rhs, eps) => {
                    let abs_diff = (lhs - rhs).abs();
                    if (abs_diff > eps) {
                        panic!("assert failed: {:?} - {:?} = {:?} <= {:?}",
                         lhs, rhs, abs_diff, eps);
                    }
                }
            }
        };
        ($v1: expr, $v2: expr) => {
            const DEFAULT_EPSILON: f32 = 0.000000001;
            assert_nearly_eq!($v1, $v2, DEFAULT_EPSILON);
        };
    }

    #[test]
    fn test_next_power_of_n_le() {
        let pow_4 = next_power_of_n_le(484, 4);
        assert_eq!(pow_4, 256);

        let pow_2 = next_power_of_n_le(72, 2);
        assert_eq!(pow_2, 64);

        let pow_2_noop = next_power_of_n_le(64, 2);
        assert_eq!(pow_2_noop, 64);
    }

    #[test]
    fn test_vec3_add() {
    {
        let a = Vec3 {x: 1.0, y: 0.0, z: 1.0};
        let b = Vec3 {x: 0.0, y: 1.0, z: 0.0};
        let c = Vec3 {x: 1.0, y: 1.0, z: 1.0};
        assert!(a + b == c);
    }
    {
        let a = Vec3 {x: 3.2, y: 0.0, z: 1.2};
        let b = Vec3 {x: 1.4, y: 0.0, z: 2.3};
        let c = Vec3 {x: 4.6, y: 0.0, z: 3.5};
        assert!(a + b == c);
    }
    }

    #[test]
    fn test_vec3_sub() {
    {
        let a = Vec3 {x: 1.0, y: 1.0, z: 1.0};
        let b = Vec3 {x: 1.0, y: 1.0, z: 1.0};
        let c = Vec3 {x: 0.0, y: 0.0, z: 0.0};
        assert!(a - b == c);
    }

    {
        let a = Vec3 {x: 5.0, y: 4.0, z: 3.0};
        let b = Vec3 {x: 2.0, y: 3.0, z: 5.5};
        let c = Vec3 {x: 3.0, y: 1.0, z: -2.5};
        assert!(a - b == c);
    }
    }

    #[test]
    fn test_vec3_length() {
    {
        let v = Vec3 {x: 1.0, y: 0.0, z: 0.0};
        assert_eq!(length(v), 1.0);
    }
    {
        let v = Vec3 {x: 3.0, y: 3.0, z: 3.0};
        assert_nearly_eq!(length(v), 5.19615242270663188058);
    }
    }

    #[test]
    fn test_vec3_length_sq()
    {
    {
        let v = Vec3 {x: 1.0, y: 0.0, z: 0.0};
        assert_nearly_eq!(length_sq(v), 1.0);
    }
    {
        let v = Vec3 {x: 3.0, y: 3.0, z: 3.0};
        assert_nearly_eq!(length_sq(v), 27.0);
    }
    }

    #[test]
    fn test_vec3_normalize()
    {
    {
        let v = Vec3 {x: 3.0, y: 3.0, z: 3.0};
        assert_nearly_eq!(length_sq(normalized(v)), 1.0, 0.00001);
    }
    {
        let v = Vec3 {x: 1.0, y: 1.0, z: 1.0};
        assert_nearly_eq!(length_sq(normalized(v)), 1.0, 0.00001);
    }
    {
        let v = Vec3 {x: 12.2, y: 5.255, z: 3.129};
        assert_nearly_eq!(length_sq(normalized(v)), 1.0, 0.00001);
    }
    }

    #[test]
    fn test_vec3_dot()
    {
    {
        let a = vec3![1.0, 0.0, 0.0];
        let b = vec3![0.0, 1.0, 0.0];
        assert_eq!(dot(a, b), 0.0);
    }
    {
        let a = normalized(vec3![0.0, 2.0, 0.0]);
        let b = normalized(vec3![0.0, 2.0, 0.0]);
        assert_eq!(dot(a, b), 1.0);
    }
    {
        let a = normalized(vec3![0.0, 0.0,  0.45]);
        let b = normalized(vec3![0.0, 0.0, -0.45]);

        assert_eq!(dot(a, b), -1.0);
    }
    }

    #[test]
    fn test_vec3_cross()
    {
    {
        let a = vec3![1.0, 0.0, 0.0];
        let b = vec3![0.0, 1.0, 0.0];
        assert_eq!(cross(a, b), vec3![0.0, 0.0, 1.0]);
    }
    {
        let a = vec3![1.0, 0.0, 0.0];
        let b = vec3![1.0, 0.0, 0.0];
        assert_eq!(cross(a, b), vec3![0.0, 0.0, 0.0]);
    }
    }
}
