use super::math::*;
use super::vec3;

#[derive(Copy,Clone)]
pub enum Axis { X, Y, Z }

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray { origin, direction }
    }

    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

pub struct HitRecord {
    pub point: Vec3,
    pub normal: Vec3,
    pub obj_id: usize,
    pub t: f32,
    pub front_face: bool    
}

impl HitRecord {
    pub fn new() -> HitRecord {
        HitRecord { point: vec3![0.0, 0.0, 0.0], normal: vec3![0.0, 0.0, 0.0], obj_id: 0, t: 0.0, front_face: false}
    }
    
    pub fn set_face_and_normal(&mut self, r: Ray, outward_normal: Vec3) {
        self.front_face = dot(r.direction, outward_normal) < 0.0;
        self.normal = if self.front_face { outward_normal } else { -outward_normal };
    }
}

#[derive(Copy, Clone)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn longest_axis(&self) -> Axis {
        let d =  self.max - self.min;
        if (d.x > d.y) && (d.x > d.z) {
            Axis::X
        }
        else if d.y > d.z {
            Axis::Y
        }
        else {
            Axis::Z
        }
    }

    fn at_axis(&self, v: &Vec3, axis: Axis) -> f32 {
        match axis {
            Axis::X => v.x,
            Axis::Y => v.y,
            Axis::Z => v.z,
        }
    }
    
    pub fn get_center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn min_at_axis(&self, axis: Axis) -> f32 {
        self.at_axis(&self.min, axis)
    }

    pub fn max_at_axis(&self, axis: Axis) -> f32 {
        self.at_axis(&self.max, axis)
    }

    pub fn merge(a: &AABB, b: &AABB) -> AABB {
        let min_x = f32::min(a.min.x, b.min.x);
        let min_y = f32::min(a.min.y, b.min.y);
        let min_z = f32::min(a.min.z, b.min.z);
        let max_x = f32::max(a.max.x, b.max.x);
        let max_y = f32::max(a.max.y, b.max.y);
        let max_z = f32::max(a.max.z, b.max.z);
        
        AABB { min: vec3![min_x, min_y, min_z], 
               max: vec3![max_x, max_y, max_z] }
    }

    pub fn hit(&self, r: Ray, mut t_min: f32, mut t_max: f32) -> bool {
        let inv_rd = 1.0 / r.direction;

        let mut t0 = (self.min.x - r.origin.x) * inv_rd.x;
        let mut t1 = (self.max.x - r.origin.x) * inv_rd.x;
        if inv_rd.x < 0.0 { std::mem::swap(&mut t0, &mut t1); }
        t_min = f32::max(t_min, t0);
        t_max = f32::min(t_max, t1);

        if t_max <= t_min { return false };

        t0 = (self.min.y - r.origin.y) * inv_rd.y;
        t1 = (self.max.y - r.origin.y) * inv_rd.y;
        if inv_rd.y < 0.0 { std::mem::swap(&mut t0, &mut t1); }
        t_min = f32::max(t_min, t0);
        t_max = f32::min(t_max, t1);

        if t_max <= t_min { return false };

        t0 = (self.min.z - r.origin.z) * inv_rd.z;
        t1 = (self.max.z - r.origin.z) * inv_rd.z;
        if inv_rd.z < 0.0 { std::mem::swap(&mut t0, &mut t1); }
        t_min = f32::max(t_min, t0);
        t_max = f32::min(t_max, t1);

        t_max > t_min
    }
}

pub struct Sphere {
    pub center: Vec3,
    pub velocity: Vec3,
    pub radius: f32,
    pub material_id: u32
}

impl Sphere {
    pub fn calc_aabb(&self) -> AABB {
        AABB {
            min: self.center - self.radius,
            max: self.center + self.radius,
        }
    }
}
