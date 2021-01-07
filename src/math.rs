// math.rs: Provides functionality for solving linear algebra problems. 

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Vec3 {
    x : f32,
    y : f32,
    z : f32
}

#[macro_export]
macro_rules! vec3 {
    ($x: expr, $y: expr, $z: expr ) => {
        Vec3 { x: $x, y: $y, z: $z}
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
    };
    // TODO(phil): Implement taking references 
}

element_wise_binary_op!(+, Add, add);
element_wise_binary_op!(-, Sub, sub);
element_wise_binary_op!(*, Mul, mul);
element_wise_binary_op!(/, Div, div);

macro_rules! element_wise_binary_op_assign {
    ($op: tt, $trait: ident, $func: ident) => {
        impl std::ops::$trait<Vec3> for &mut Vec3 {
            fn $func(&mut self, rhs: Vec3) {
                self.x $op rhs.x;
                self.y $op rhs.y;
                self.z $op rhs.z;
            }
        }

        impl std::ops::$trait<f32> for &mut Vec3 {
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

impl Vec3 {
    fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    // TODO(phil): Ideally I'd like to take a mut ref to self, get the len and then div_assign to self. 
    // When I tried, I ended up with horrible syntax, I think I dont understand lifetimes well enough 
    // for this yet. Come back, change and profile once I do.
    fn normalized(&self) -> Vec3 {
        let inv_len = 1.0 / self.length();
        Vec3 {x: self.x * inv_len, y: self.y * inv_len, z: self.z * inv_len}
    }

    fn dot(&self, b: &Vec3) -> f32 {
        self.x * b.x + self.y * b.y + self.z * b.z
    }

    fn cross(&self, b: &Vec3) -> Vec3 {
        let a = self;
        Vec3 {
            x: (a.y * b.z - a.z * b.y),
            y: -(a.x * b.z - a.z * b.x),
            z: (a.x * b.y - a.y * b.x)
        }
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
        assert_eq!(v.length(), 1.0);
    }
    {
        let v = Vec3 {x: 3.0, y: 3.0, z: 3.0};
        assert_nearly_eq!(v.length(), 5.19615242270663188058);
    }
    }

    #[test]
    fn test_vec3_length_sq()
    {
    {
        let v = Vec3 {x: 1.0, y: 0.0, z: 0.0};
        assert_nearly_eq!(v.length_sq(), 1.0);
    }
    {
        let v = Vec3 {x: 3.0, y: 3.0, z: 3.0};
        assert_nearly_eq!(v.length_sq(), 27.0);
    }
    }

    #[test]
    fn test_vec3_normalize()
    {
    {
        let v = Vec3 {x: 3.0, y: 3.0, z: 3.0};
        assert_nearly_eq!(v.normalized().length_sq(), 1.0, 0.00001);
    }
    {
        let v = Vec3 {x: 1.0, y: 1.0, z: 1.0};
        assert_nearly_eq!(v.normalized().length_sq(), 1.0, 0.00001);
    }
    {
        let v = Vec3 {x: 12.2, y: 5.255, z: 3.129};
        assert_nearly_eq!(v.normalized().length_sq(), 1.0, 0.00001);
    }
    }

    #[test]
    fn test_vec3_dot()
    {
    {
        let a = vec3![1.0, 0.0, 0.0];
        let b = vec3![0.0, 1.0, 0.0];
        assert_eq!(a.dot(&b), 0.0);
    }
    {
        let a = vec3![0.0, 2.0, 0.0].normalized();
        let b = vec3![0.0, 2.0, 0.0].normalized();
        assert_eq!(a.dot(&b), 1.0);
    }
    {
        let a = vec3![0.0, 0.0,  0.45].normalized();
        let b = vec3![0.0, 0.0, -0.45].normalized();

        assert_eq!(a.dot(&b), -1.0);
    }
    }

    #[test]
    fn test_vec3_cross()
    {
    {
        let a = vec3![1.0, 0.0, 0.0];
        let b = vec3![0.0, 1.0, 0.0];
        assert_eq!(a.cross(&b), vec3![0.0, 0.0, 1.0]);
    }
    {
        let a = vec3![1.0, 0.0, 0.0];
        let b = vec3![1.0, 0.0, 0.0];
        assert_eq!(a.cross(&b), vec3![0.0, 0.0, 0.0]);
    }
    }
}