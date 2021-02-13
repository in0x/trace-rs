#![feature(stdsimd)]
#![feature(platform_intrinsics)]
#![feature(link_llvm_intrinsics)]
#![feature(simd_ffi)]

pub mod math;
pub mod simd;

use math::*;
use simd::*;
use std::time::Instant;

extern crate image;
extern crate random_fast_rng;

pub struct Camera {
    eye_origin: Vec3,
    lower_left: Vec3,
    x_extent: Vec3,
    y_extent: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f32
}

impl Camera {
    pub fn new(origin: Vec3, look_at: Vec3, up: Vec3, y_fov_deg: f32, aspect_ratio: f32, aperture: f32, focus_dist: f32) -> Camera {
        let theta = degree_to_rad(y_fov_deg);
        let h = f32::tan(theta / 2.0);
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;
        
        let w = normalized(origin - look_at);
        let u = normalized(cross(up, w));
        let v = cross(w, u);

        let horizontal = focus_dist * viewport_width * u;
        let vertical = focus_dist * viewport_height * v;

        Camera {
            eye_origin: origin,
            lower_left: origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w,
            x_extent: horizontal,
            y_extent: vertical,
            u, v,
            lens_radius: aperture / 2.0
        }
    }

    pub fn get_ray(&self, s: f32, t: f32, time0: f32, time1: f32) -> Ray {
        let rd = self.lens_radius * rand_in_unit_disk();
        let offset = self.u * rd.x + self.v * rd.y;

        Ray::new(
            self.eye_origin + offset, 
            self.lower_left + s * self.x_extent + t * self.y_extent - self.eye_origin - offset,
            rand_in_range(time0, time1)
        )
    }
}

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub time: f32
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3, time: f32) -> Ray {
        Ray { origin, direction, time }
    }

    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

struct HitRecord {
    point: Vec3,
    normal: Vec3,
    obj_id: usize,
    t: f32,
    front_face: bool    
}

impl HitRecord {
    fn new() -> HitRecord {
        HitRecord { point: vec3![0.0, 0.0, 0.0], normal: vec3![0.0, 0.0, 0.0], obj_id: 0, t: 0.0, front_face: false}
    }
    
    fn set_face_and_normal(&mut self, r: Ray, outward_normal: Vec3) {
        self.front_face = dot(r.direction, outward_normal) < 0.0;
        self.normal = if self.front_face { outward_normal } else { -outward_normal };
    }
}

// unsafe fn hmax(mut m: __m128) -> f32 {
//     m = _mm_max_ps(m, MY_SHUFFLE_4!(m, 2, 3, 0, 0));
//     m = _mm_max_ps(m, MY_SHUFFLE_4!(m, 1, 0, 0, 0));
//     _mm_cvtss_f32(m)
// }

const SIMD_WIDTH: usize = 4;

fn hit_simd(world: &World, start: usize, len: usize, r: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool {
    let ro_x = f32x4::splat(r.origin.x);
    let ro_y = f32x4::splat(r.origin.y);
    let ro_z = f32x4::splat(r.origin.z);
    let rd_x = f32x4::splat(r.direction.x);
    let rd_y = f32x4::splat(r.direction.y);
    let rd_z = f32x4::splat(r.direction.z);
    let r_t  = f32x4::splat(r.time);

    let t_min4 = f32x4::splat(t_min);
    let zero_4 = f32x4::splat(0.0);

    let mut id = i32x4::splat(-1);
    let mut closest_t = f32x4::splat(t_max);

    let first_idx = start as i32;
    let mut cur_id = i32x4::set(first_idx, first_idx + 1, first_idx + 2, first_idx + 3);

    let end = start + len;
    let sphere_count = end - (end % SIMD_WIDTH) - SIMD_WIDTH;
    for i in (start..sphere_count).step_by(SIMD_WIDTH) {
        let vel_x =  f32x4::loadu(&world.animations.velocity_x[i..i+4]);
        let vel_y =  f32x4::loadu(&world.animations.velocity_y[i..i+4]);
        let vel_z =  f32x4::loadu(&world.animations.velocity_z[i..i+4]);
        let time_0 = f32x4::loadu(&world.animations.start_time[i..i+4]);
        let time_1 = f32x4::loadu(&world.animations.end_time[i..i+4]);

        let time_t = f32x4::div(f32x4::sub(r_t, time_0), f32x4::sub(time_1, time_0));
    
        let sc_x = f32x4::add(f32x4::loadu(&world.sphere_x[i..i+4]), f32x4::mul(vel_x, time_t));
        let sc_y = f32x4::add(f32x4::loadu(&world.sphere_y[i..i+4]), f32x4::mul(vel_y, time_t));
        let sc_z = f32x4::add(f32x4::loadu(&world.sphere_z[i..i+4]), f32x4::mul(vel_z, time_t));

        let _radius =  f32x4::loadu(&world.sphere_r[i..i+4]);
        let sq_radius = f32x4::mul(_radius, _radius);

        let co_x = f32x4::sub(sc_x, ro_x);
        let co_y = f32x4::sub(sc_y, ro_y);
        let co_z = f32x4::sub(sc_z, ro_z);

        // Note: nb is minus b, avoids having to negate b (and the sign cancels out in c).
        let a  = f32x4::add(f32x4::add(f32x4::mul(rd_x, rd_x), f32x4::mul(rd_y, rd_y)), f32x4::mul(rd_z, rd_z)); 
        let nb = f32x4::add(f32x4::add(f32x4::mul(co_x, rd_x), f32x4::mul(co_y, rd_y)), f32x4::mul(co_z, rd_z));
        let c  = f32x4::sub(f32x4::add(f32x4::add(f32x4::mul(co_x, co_x), f32x4::mul(co_y, co_y)), f32x4::mul(co_z, co_z)), sq_radius);

        let discr = f32x4::sub(f32x4::mul(nb, nb), f32x4::mul(a, c));
        let discr_gt_0 = f32x4::cmp_gt(discr, zero_4);

        // See if any of the four spheres have a solution.
        if discr_gt_0.any() {
            let discr_root = f32x4::sqrt(discr);
            let t0 = f32x4::div(f32x4::sub(nb, discr_root), a);
            let t1 = f32x4::div(f32x4::add(nb, discr_root), a);

            let best_t = f32x4::blendv(t0, t1, f32x4::cmp_gt(t0, t_min4)); // If t0 is above min, take it (since it's the earlier hit), else try t1.
            let mask = u32x4::and(u32x4::and(discr_gt_0, f32x4::cmp_gt(best_t, t_min4)), f32x4::cmp_lt(best_t, closest_t));
            
            // if hit, take it
            id = i32x4::blendv(cur_id, id, mask);
            closest_t = f32x4::blendv(best_t, closest_t, mask);
        }
        
        cur_id = i32x4::add(cur_id, i32x4::splat(SIMD_WIDTH as i32));
    }
    
    // We found the four best hits, lets see if any are actually close.
    let min_t = f32x4::hmin(closest_t);
    if min_t < t_max { 
        let min_t_channel = f32x4::cmp_eq(closest_t, f32x4::splat(min_t));
        let min_t_mask = u32x4::vmovemaskq_u32(min_t_channel.m);
        if min_t_mask != 0 {
            let mut id_scalar : [i32;4] = [0, 0, 0, 0];
            let mut closest_t_scalar : [f32;4] = [0.0, 0.0, 0.0, 0.0];

            i32x4::storeu(id, &mut id_scalar);
            f32x4::storeu(closest_t, &mut closest_t_scalar);

            // Map the lowest set bit of the value indexed with. 
            let mask_to_channel : [i32;16] =
            [
                0, 0, 1, 0, // 00xx
                2, 0, 1, 0, // 01xx
                3, 0, 1, 0, // 10xx
                2, 0, 1, 0, // 11xx
            ];

            let lane = mask_to_channel[min_t_mask as usize];
            let hit_id = id_scalar[lane as usize] as usize;
            let final_t = closest_t_scalar[lane as usize];

            out_hit.point = r.at(final_t);
            out_hit.normal = (out_hit.point - world.sphere_center(hit_id, r.time)) / world.sphere_r[hit_id];
            out_hit.set_face_and_normal(r, out_hit.normal);
            out_hit.t = final_t;
            out_hit.obj_id = hit_id as usize;
            return true;
        }
    }
    
    false
}

fn hit(world: &World, start: usize, len: usize, ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool {
    let mut closest_t = t_max;
    let mut hit_id = -1;

    let end = start + len;
    for i in start..end {
        let center = world.sphere_center(i, ray.time);
        let radius = world.sphere_r[i];

        let oc = ray.origin - center;
        let a = dot(ray.direction, ray.direction);
        let b = dot(oc, ray.direction);
        let c = dot(oc, oc) - radius * radius;

        let discriminant = b*b - a*c; 
        if discriminant > 0.0 {
            let d_root = discriminant.sqrt();
            let t1 = (-b - d_root) / a;
            let t2 = (-b + d_root) / a;

            if t1 < closest_t && t1 > t_min {
                closest_t = t1;
                hit_id = i as i32;
            } 
            else if t2 < closest_t && t2 > t_min {
                closest_t = t2;
                hit_id = i as i32;
            }
        }       
    }
    
    if hit_id != -1 {
        let center = world.sphere_center(hit_id as usize, ray.time);
        let radius = world.sphere_r[hit_id as usize];

        out_hit.t = closest_t;
        out_hit.obj_id = hit_id as usize;
        out_hit.point = ray.at(closest_t);
        let outward_normal = (out_hit.point - center) / radius;
        out_hit.set_face_and_normal(ray, outward_normal);
        true    
    } 
    else {
        false
    }
}    

trait Hitable {
    fn hit(&self, ray: Ray, t_min: f32, t_max: f32, world: &World, out_hit: &mut HitRecord) -> bool;
}

fn hit_spheres(world: &World, start: usize, len: usize, ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool {
    let mut found_hit = false;
    let mut closest_t = t_max;

    if len >= SIMD_WIDTH {
        if hit_simd(world, start, len, ray, t_min, closest_t, out_hit) {
            found_hit = true;
            closest_t = out_hit.t;
        }
        
        let end = start + len;
        found_hit |= hit(world, end - (end % SIMD_WIDTH), end % SIMD_WIDTH, ray, t_min, closest_t, out_hit);
    }
    else {
        found_hit = hit(world, start, len, ray, t_min, closest_t, out_hit);
    }
    
    found_hit
}

// fn hit_world(r: Ray, t_min: f32, t_max: f32, current_node: &BVHNode, bvh: &BVH, world: &World, out_hit: &mut HitRecord) -> bool {        
//     if !current_node.bounds.hit(r, t_min, t_max) {
//         return false;
//     }

//     if current_node.geometry_idx != INVALID_GEO_IDX {
//         let start = current_node.geometry_idx as usize;
//         let len = current_node.geometry_len as usize;
//         return hit_spheres(world, start, len, r, t_min, t_max, out_hit);
//     }

//     let l_node = &bvh.tree[current_node.left_child as usize];
//     let r_node = &bvh.tree[current_node.right_child as usize];

//     let l_hit = hit_world(r, t_min, t_max, l_node, bvh, world, out_hit);
//     let closest_t = if l_hit { out_hit.t } else { t_max }; 
//     let r_hit = hit_world(r, t_min, closest_t, r_node, bvh, world, out_hit);

//     l_hit || r_hit
// }

fn sample_color(r: Ray, hit_tree: & dyn Hitable, world: &World, materials: &[Material], bounces: i32) -> Vec3 {
    let mut hit = HitRecord::new();

    let mut color = vec3![1.0, 1.0, 1.0];
    let mut current_ray = r;

    for _bounce in 0..bounces {
        // if hit_world(current_ray, 0.001, f32::MAX, &bvh.root, bvh, &world, &mut hit) {
        // if hit_spheres(world, 0, world.sphere_x.len(), current_ray, 0.001, f32::MAX, &mut hit) {
        if hit_tree.hit(current_ray, 0.001, f32::MAX, world, &mut hit) {
            hit.normal = normalized(hit.normal); // The book makes a point about unit length normals but then doesnt enforce it.
            let mat_id = world.material_ids[hit.obj_id] as usize;
            let (did_bounce, attenuation, scattered) = materials[mat_id].scatter(&current_ray, &hit);
            if did_bounce {
                current_ray = scattered;
                color *= attenuation;
            }
        } else {
            color *= get_sky_color(current_ray);
            break;
        }
    }

    color
}

fn get_sky_color(r: Ray) -> Vec3 {
    let unit_dir = normalized(r.direction);
    let t = 0.5 * (unit_dir.y + 1.0);
    (1.0 - t) * vec3![1.0, 1.0, 1.0] + t * vec3![0.5, 0.7, 1.0]    
}

trait Scattering {
    /// Returns a triplet of (did_hit, attenutation, scattered_ray).
    fn scatter(&self, r: &Ray, hit: &HitRecord) -> (bool, Vec3, Ray);
}

struct Lambert {
    albedo: Vec3
}

impl Scattering for Lambert {
    fn scatter(&self, r: &Ray, hit: &HitRecord) -> (bool, Vec3, Ray) {
        let mut scatter_dir = hit.normal + rand_unit_vector();
        if scatter_dir.is_near_zero() {
            scatter_dir = hit.normal;
        }

        let scattered = Ray::new(hit.point, scatter_dir, r.time);
        let attenuation = self.albedo;
        (true, attenuation, scattered)
    }
}

struct Metal {
    albedo: Vec3,
    roughness: f32 
}

impl Metal {
    fn new(albedo: Vec3, roughness: f32) -> Metal {
        Metal { albedo, roughness: f32::min(roughness, 1.0) }
    }
}

impl Scattering for Metal {
    fn scatter(&self, r: &Ray, hit: &HitRecord,) -> (bool, Vec3, Ray) {
        let reflected = reflect(normalized(r.direction), hit.normal);
        let scattered = Ray::new(hit.point, reflected + self.roughness * rand_in_unit_sphere(), r.time);
        let attenuation = self.albedo;
        let did_bounce = dot(scattered.direction, hit.normal) > 0.0;
        (did_bounce, attenuation, scattered)
    }
}

struct Dielectric {
    idx_of_refraction : f32
}

fn schlick_reflectance(cosine: f32, ref_idx: f32) -> f32 {
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * f32::powi(1.0 - cosine, 5)
}

impl Scattering for Dielectric {
    fn scatter(&self, r: &Ray, hit: &HitRecord) -> (bool, Vec3, Ray) {
        let refraction_ratio = if hit.front_face {1.0 / self.idx_of_refraction} else { self.idx_of_refraction };
        
        let unit_dir = normalized(r.direction);
        let cos_theta = f32::min(dot(-unit_dir, hit.normal), 1.0);
        let sin_theta = f32::sqrt(1.0 - cos_theta * cos_theta);

        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let should_reflect = schlick_reflectance(cos_theta, refraction_ratio) > rand_f32();
        let direction = match cannot_refract || should_reflect {
            true => reflect(unit_dir, hit.normal),
            false => refract(unit_dir, hit.normal, refraction_ratio)
        };

        (true, vec3![1.0], Ray::new(hit.point, direction, r.time))
    }
}

enum Material {
    Lambert(Lambert),
    Metal(Metal),
    Dielectric(Dielectric),
}

impl Scattering for Material {
    fn scatter(&self, r: &Ray, hit: &HitRecord) -> (bool, Vec3, Ray) {
        match self {
            Material::Lambert(mat) => mat.scatter(r, hit),
            Material::Metal(mat) => mat.scatter(r, hit),
            Material::Dielectric(mat) => mat.scatter(r, hit),
        }
    }
}

#[derive(Default)]
struct AnimationData {
    velocity_x: Vec<f32>,
    velocity_y: Vec<f32>,
    velocity_z: Vec<f32>,
    start_time: Vec<f32>,
    end_time:   Vec<f32>,
}

impl AnimationData {
    fn get_velocity(&self, idx: usize) -> Vec3 {
        vec3![self.velocity_x[idx], self.velocity_y[idx], self.velocity_z[idx]]
    }
}


#[derive(Default)]
struct World {
    sphere_x: Vec<f32>,
    sphere_y: Vec<f32>,
    sphere_z: Vec<f32>,
    sphere_r: Vec<f32>,
    material_ids: Vec<u32>,
    animations: AnimationData,
}

impl World {
    fn push_moving_sphere(&mut self, center: Vec3, radius: f32, material_id: u32, vel: Vec3, start_time: f32, end_time: f32) {
        self.sphere_x.push(center.x);
        self.sphere_y.push(center.y);
        self.sphere_z.push(center.z);
        self.sphere_r.push(radius);
        self.material_ids.push(material_id);
        self.animations.velocity_x.push(vel.x);
        self.animations.velocity_y.push(vel.y);
        self.animations.velocity_z.push(vel.z);
        self.animations.start_time.push(start_time);
        self.animations.end_time.push(end_time);
    }

    fn sphere_center(&self, idx: usize, time: f32) -> Vec3 {
        // NOTE(): Could have is_static flag to shortcut calculations here. 
        let center = vec3![self.sphere_x[idx], self.sphere_y[idx], self.sphere_z[idx]];

        let velocity =  self.animations.get_velocity(idx);
        let time0 = self.animations.start_time[idx];
        let time1 = self.animations.end_time[idx];

        center + ((time - time0) / (time1 - time0)) * velocity 
    }

    fn construct(objects: &[Sphere]) -> World {
        let mut world = World::default();
        for sphere in objects {
            world.push_moving_sphere(
                sphere.center,
                sphere.radius,
                sphere.material_id,
                sphere.velocity,
                sphere.start_time,
                sphere.end_time,
            );
        }
        world
    }
}

struct Sphere {
    center: Vec3,
    velocity: Vec3,
    start_time: f32,
    end_time: f32,
    radius: f32,
    material_id: u32
}

impl Sphere {
    fn calc_aabb(&self, time0: f32, time1: f32) -> AABB {
        let c0 = self.get_center(time0);
        let c1 = self.get_center(time1);
        
        let bb0 = AABB {
            min: c0 - self.radius,
            max: c0 - self.radius,
        };

        let bb1 = AABB {
            min: c1 - self.radius,
            max: c1 + self.radius,
        };

        AABB::merge(&bb0, &bb1)
    }

    fn get_center(&self, time: f32) -> Vec3 {
        self.center + ((time - self.start_time) / (self.end_time - self.start_time)) * self.velocity  
    }
}

fn push_sphere(objects: &mut Vec<Sphere>, center: Vec3, radius: f32, mat: usize) {
    objects.push(Sphere {
        center, radius, 
        material_id: mat as u32,
        velocity: Vec3::zero(),
        start_time: 0.0,
        end_time: 1.0
    });
}

fn push_moving_sphere(objects: &mut Vec<Sphere>, center: Vec3, radius: f32, mat: usize, 
    vel: Vec3, start_time: f32, end_time: f32) {
    objects.push(Sphere {
        center, radius, 
        material_id: mat as u32,
        velocity: vel,
        start_time,
        end_time
    });
}

fn generate_world_data(objects: &mut Vec<Sphere>, materials: &mut Vec<Material>) {

    materials.push(Material::Lambert(Lambert{ albedo: vec3![0.5, 0.5, 0.5] }));
    push_sphere(objects, vec3![0.0, -1000.0, 0.0], 1000.0, materials.len() - 1);

    for a in -11..11 {
        for b in -11..11 {
            let material_select : f32 = rand_f32();
            let rand_a : f32 = rand_f32();
            let rand_b : f32 = rand_f32();
            let center = vec3![(a as f32) + 0.9 * rand_a, 0.2, (b as f32) + 0.9 * rand_b];
            
            if length(center - vec3![4.0, 0.2, 0.0]) > 0.9 {
                if material_select < 0.8 {
                    let albedo = Vec3::random() * Vec3::random();
                    let velocity = vec3![0.0, rand_in_range(0.0, 0.5), 0.0];
                    materials.push(Material::Lambert(Lambert{ albedo }));
                    push_moving_sphere(objects, center, 0.2, materials.len() - 1, velocity, 0.0, 1.0);
                } 
                else if material_select < 0.96 {
                    let albedo = Vec3::random_range(0.5, 1.0);
                    let rough = rand_in_range(0.0, 0.5);
                    materials.push(Material::Metal(Metal::new(albedo, rough)));
                    push_sphere(objects, center, 0.2, materials.len() - 1);
                } 
                else {
                    materials.push(Material::Dielectric(Dielectric{ idx_of_refraction: 1.5 }));
                    push_sphere(objects, center, 0.2, materials.len() - 1);
                }
            }
        }
    }

    materials.push(Material::Dielectric(Dielectric{ idx_of_refraction: 1.5 }));
    push_sphere(objects, vec3![0.0, 1.0, 0.0], 1.0, materials.len() - 1);
    
    materials.push(Material::Lambert(Lambert{ albedo: vec3![0.4, 0.2, 0.1] }));
    push_sphere(objects, vec3![-4.0, 1.0, 0.0], 1.0, materials.len() - 1);
    
    materials.push(Material::Metal(Metal{ albedo: vec3![0.7, 0.6, 0.5], roughness: 0.0 }));
    push_sphere(objects, vec3![4.0, 1.0, 0.0], 1.0, materials.len() - 1);
}

#[derive(Copy, Clone)]
struct AABB {
    min: Vec3,
    max: Vec3,
}

impl AABB {
    fn new() -> AABB {
        AABB { min: Vec3::zero(), max: Vec3::zero() }
    }

    fn is_empty(&self) -> bool {
        length_sq(self.min) == 0.0 && length_sq(self.max) == 0.0
    }

    fn merge(a: &AABB, b: &AABB) -> AABB {
        let min_x = f32::min(a.min.x, b.min.x);
        let min_y = f32::min(a.min.y, b.min.y);
        let min_z = f32::min(a.min.z, b.min.z);
        let max_x = f32::max(a.max.x, b.max.x);
        let max_y = f32::max(a.max.y, b.max.y);
        let max_z = f32::max(a.max.z, b.max.z);
        
        AABB { min: vec3![min_x, min_y, min_z], 
               max: vec3![max_x, max_y, max_z] }
    }

    fn hit(&self, r: Ray, mut t_min: f32, mut t_max: f32) -> bool {
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

    // unsafe fn hit_wide(r: Ray, t_min: f32, t_max: f32) -> bool {
    //     let min_x = _mm_set_ps1(0.0);
    //     let min_y = _mm_set_ps1(0.0);
    //     let min_z = _mm_set_ps1(0.0);
        
    //     let max_x = _mm_set_ps1(0.0);
    //     let max_y = _mm_set_ps1(0.0);
    //     let max_z = _mm_set_ps1(0.0);
    
    //     let ro_x = _mm_set_ps1(r.origin.x);
    //     let ro_y = _mm_set_ps1(r.origin.y);
    //     let ro_z = _mm_set_ps1(r.origin.z);
    
    //     let inv_rd_x = _mm_set_ps1(1.0 / r.direction.x);
    //     let inv_rd_y = _mm_set_ps1(1.0 / r.direction.y);
    //     let inv_rd_z = _mm_set_ps1(1.0 / r.direction.z);
    
    //     let zero_4 = _mm_set_ps1(0.0);
    //     let t_min_4 = _mm_set_ps1(t_min);
    //     let t_max_4 = _mm_set_ps1(t_max);
        
    //     let mut t0_x = _mm_mul_ps(_mm_sub_ps(min_x, ro_x), inv_rd_x);
    //     let mut t1_x = _mm_mul_ps(_mm_sub_ps(max_x, ro_x), inv_rd_x);
    
    //     let inv_rd_x_lt0 = _mm_cmplt_ps(inv_rd_x, zero_4);
    //     t0_x = _mm_blendv_ps(t0_x, t1_x, inv_rd_x_lt0);
    //     t1_x = _mm_blendv_ps(t1_x, t0_x, inv_rd_x_lt0);
    
    //     t0_x = _mm_max_ps(t_min_4, t0_x);
    //     t1_x = _mm_min_ps(t_max_4, t1_x);
    
    //     let any_x_hit = _mm_cmple_ps(t1_x, t0_x);
    //     if _mm_movemask_ps(any_x_hit) != 0 {
    //         // TODO(): lane select
    //         return true;
    //     }
    
    //     // TODO(): repeat for y and z
    
    //     return false;
    // }

    // fn _hit_simd(&self, r: Ray, t_min: f32, t_max: f32) -> bool {
    // unsafe {
    //         let inv_rd = _mm_div_ps(_mm_set_ps1(1.0), _mm_set_ps(r.direction.x, r.direction.y, r.direction.z, 1.0));
    //         let ro = _mm_set_ps(r.origin.x, r.origin.y, r.origin.z, 0.0);
    
    //         let bb_min = _mm_set_ps(self.min.x, self.min.y, self.min.z, 0.0);
    //         let bb_max = _mm_set_ps(self.max.x, self.max.y, self.max.z, 0.0);
    
    //         let t0 = _mm_mul_ps(_mm_sub_ps(bb_min, ro), inv_rd);
    //         let t1 = _mm_mul_ps(_mm_sub_ps(bb_max, ro), inv_rd);
    
    //         let inv_lt_0 = _mm_cmplt_ps(inv_rd, _mm_set_ps1(0.0));

    //         let mut swap_min = _mm_blendv_ps(t0, t1, inv_lt_0);
    //         let mut swap_max = _mm_blendv_ps(t1, t0, inv_lt_0);

    //         swap_min = _mm_blend_ps(swap_min, _mm_set_ps1(t_min), 1);
    //         swap_max = _mm_blend_ps(swap_max, _mm_set_ps1(t_max), 1);

    //         hmin(swap_max) > hmax(swap_min)
    // }
    // }
}

struct BVHNode {
    bounds: AABB,
    left_child: u32,
    right_child: u32,
    geometry_idx: i32,
    geometry_len: u32
}

const INVALID_GEO_IDX: i32 = -1;

impl BVHNode {
    fn new() -> BVHNode {
        BVHNode {
            bounds: AABB::new(), left_child: 0, right_child: 0, 
            geometry_idx: INVALID_GEO_IDX, geometry_len: 0
        }
    }
}

struct BVH {
    root: BVHNode,
    tree: Vec<BVHNode>
}

impl BVH {
    fn hit_world(r: Ray, t_min: f32, t_max: f32, current_node: &BVHNode, bvh: &BVH, world: &World, out_hit: &mut HitRecord) -> bool {        
        if !current_node.bounds.hit(r, t_min, t_max) {
            return false;
        }

        if current_node.geometry_idx != INVALID_GEO_IDX {
            let start = current_node.geometry_idx as usize;
            let len = current_node.geometry_len as usize;
            return hit_spheres(world, start, len, r, t_min, t_max, out_hit);
        }

        let l_node = &bvh.tree[current_node.left_child as usize];
        let r_node = &bvh.tree[current_node.right_child as usize];

        let l_hit = BVH::hit_world(r, t_min, t_max, l_node, bvh, world, out_hit);
        let closest_t = if l_hit { out_hit.t } else { t_max }; 
        let r_hit = BVH::hit_world(r, t_min, closest_t, r_node, bvh, world, out_hit);

        l_hit || r_hit
    }
}

impl Hitable for BVH {
    fn hit(&self, ray: Ray, t_min: f32, t_max: f32, world: &World, out_hit: &mut HitRecord) -> bool {
        return BVH::hit_world(ray, t_min, t_max, &self.root, &self, world, out_hit);
    }
}

#[derive(Copy,Clone)]
enum Axis { X, Y, Z }

fn random_axis() -> Axis {
    let mut rand_axis: u32 = rand_int();
    rand_axis %= 3;
    
    match rand_axis {
        0 => Axis::X,
        1 => Axis::Y,
        2 => Axis::Z,
        _ => unreachable!("Unexpected axis value")
    }
}

fn calc_aabb(objects: &[Sphere], time0: f32, time1: f32) -> AABB {
    let mut bb = objects[0].calc_aabb(time0, time1);
    for sphere in objects {
        bb = AABB::merge(&bb, &sphere.calc_aabb(time0, time1))
    }
    bb
}

fn sort(objects: &mut [Sphere], axis: Axis, time0: f32, time1: f32) {
    match axis { // TODO(): this can be written a lot more concisely
        Axis::X => {
            objects.sort_by(|lhs, rhs| {
                let bb_lhs = lhs.calc_aabb(time0, time1); let bb_rhs = rhs.calc_aabb(time0, time1);
                bb_lhs.min.x.partial_cmp(&bb_rhs.min.x).unwrap()
            });
        },
        Axis::Y => {
            objects.sort_by(|lhs, rhs| {
                let bb_lhs = lhs.calc_aabb(time0, time1); let bb_rhs = rhs.calc_aabb(time0, time1);
                bb_lhs.min.y.partial_cmp(&bb_rhs.min.y).unwrap()
            });
        },
        Axis::Z => {
            objects.sort_by(|lhs, rhs| {
                let bb_lhs = lhs.calc_aabb(time0, time1); let bb_rhs = rhs.calc_aabb(time0, time1);
                bb_lhs.min.z.partial_cmp(&bb_rhs.min.z).unwrap()
            });
        }
    }
}

fn construct_bvh(objects: &mut[Sphere], time0: f32, time1: f32, bvh: &mut Vec<BVHNode>, cur_node: &mut BVHNode, start: usize, end: usize) {    
    let split_axis = random_axis();
    let range = end - start;

    let sub_objects = &mut objects[start..end];

    if range <= SIMD_WIDTH * 8 {
        cur_node.geometry_idx = start as i32;
        cur_node.geometry_len = range as u32;
        cur_node.bounds = calc_aabb(sub_objects, time0, time1);
        return;
    }
    else {
        sort(sub_objects, split_axis, time0, time1);

        let mid = start + range / 2;
        
        let mut l_node = BVHNode::new();
        cur_node.left_child = bvh.len() as u32;
        bvh.push(BVHNode::new());
        construct_bvh(objects, time0, time1, bvh, &mut l_node, start, mid);
        bvh[cur_node.left_child as usize] = l_node;
        
        let mut r_node = BVHNode::new();
        cur_node.right_child = bvh.len() as u32;
        bvh.push(BVHNode::new());
        construct_bvh(objects, time0, time1, bvh, &mut r_node, mid, end);
        bvh[cur_node.right_child as usize] = r_node;
    }

    let left_bb = bvh[cur_node.left_child as usize].bounds;
    assert!(!left_bb.is_empty());

    let right_bb = bvh[cur_node.right_child as usize].bounds;
    assert!(!right_bb.is_empty());

    cur_node.bounds = AABB::merge(&left_bb, &right_bb);
}

enum BVHNodeType {
    Joint, /// This node's children are BVHNodes.
    Leaf,  /// This node's children are intersectibles.
    EnumCount
}

/// A QBVHNode has up to four children that it stores the bounds of.
/// This allows fast intersection testing by testing all 4 at once through SIMD.
struct QBVHNode {
    bb_min_x: [f32; 4],
    bb_min_y: [f32; 4],
    bb_min_z: [f32; 4],
    bb_max_x: [f32; 4],
    bb_max_y: [f32; 4],
    bb_max_z: [f32; 4],
    children: [i32; 4],
    child_type: [BVHNodeType; 4],
    num_children: u8
}

impl QBVHNode {
    fn new() -> QBVHNode {
        QBVHNode {
            bb_min_x: [0.0, 0.0, 0.0, 0.0],
            bb_min_y: [0.0, 0.0, 0.0, 0.0],
            bb_min_z: [0.0, 0.0, 0.0, 0.0],
            bb_max_x: [0.0, 0.0, 0.0, 0.0],
            bb_max_y: [0.0, 0.0, 0.0, 0.0],
            bb_max_z: [0.0, 0.0, 0.0, 0.0],
            children: [-1, -1, -1, -1],
            child_type: [BVHNodeType:: Joint, BVHNodeType:: Joint, BVHNodeType:: Joint, BVHNodeType:: Joint],
            num_children: 0
            // ^ We want to indicate what the children are, so we can avoid inserting
            // BVHNodes for single object children since that just wastes space.
        }
    }
}

struct QBVH {
    root: QBVHNode,
    tree: Vec<QBVHNode>
}

/// Retrieves the bounds for the child of node at child_idx.
fn get_qbvh_aabb(node: &QBVHNode, child_idx: u8) -> AABB {
    let i = child_idx as usize;
    AABB {
        min: vec3![node.bb_min_x[i], node.bb_min_y[i], node.bb_min_z[i]],
        max: vec3![node.bb_max_x[i], node.bb_max_y[i], node.bb_max_z[i]],
    }
}

/// Sets the merged bounds on the node for its child at child_idx.
fn set_qbvh_aabb(node: &mut QBVHNode, child_idx: u8, bb: &AABB) {
    node.bb_min_x[child_idx as usize] = bb.min.x;
    node.bb_min_y[child_idx as usize] = bb.min.y;
    node.bb_min_z[child_idx as usize] = bb.min.z;
    node.bb_max_x[child_idx as usize] = bb.max.x;
    node.bb_max_y[child_idx as usize] = bb.max.y;
    node.bb_max_z[child_idx as usize] = bb.max.z;   
}

/// Calculates the merged bounds of child of this node and assigns it 
/// to the appropriate slot in this node.
fn merge_bounds_from_children(node: &mut QBVHNode, qbvh: &Vec<QBVHNode>) {
    for child_i in 0..node.num_children {

        let child_node = &qbvh[child_i as usize];
        assert_ne!(child_node.num_children, 0);

        let mut bounds = get_qbvh_aabb(&child_node, 0);
        for node_i in 1..child_node.num_children {
            bounds = AABB::merge(&bounds, &get_qbvh_aabb(&child_node, node_i));
        }

        set_qbvh_aabb(node, child_i, &bounds);
    }
}


fn construct_qbvh(objects: &mut[Sphere], t0: f32, t1: f32, bvh: &mut Vec<QBVHNode>, cur_node: &mut QBVHNode, start: usize, end: usize) {

    let split_axis = random_axis();
    let range = end - start;

    let sub_objects = &mut objects[start..end];

    if range <= 4 {

        for i in 0..range {          
            let aabb = objects[i].calc_aabb(t0, t1);
            cur_node.bb_min_x[i] = aabb.min.x;
            cur_node.bb_min_y[i] = aabb.min.y;
            cur_node.bb_min_z[i] = aabb.min.z;
            cur_node.bb_max_x[i] = aabb.max.x;
            cur_node.bb_max_y[i] = aabb.max.y;
            cur_node.bb_max_z[i] = aabb.max.z;
            cur_node.child_type[i] = BVHNodeType::Leaf;
            cur_node.children[i] = (start + i) as i32;
            cur_node.num_children += 1;
        }

        // pad if we dont have 4 nodes to fill with
        // TODO(): we should probably assert on this in the hit logic. The bvh
        // for this child will be all 0, so we should never hit it and get an idx of -1.
        for i in range..4 {
            cur_node.children[i] = -1;
            cur_node.child_type[i] = BVHNodeType::Leaf;
        }
    }
    // TODO(): Do we insert a case here we handle setting up leaf nodes?
    // Or we could replace the above case with one that checks the next divide.
    else if range < 16 {
        // next nodes will end up being leaves, should pad to four
        // or should we? is full 4 prefereable? Probably, better simd occupancy

        let mut start_idx = start as i32;
        let mut nodes_to_insert = range as i32;
        while nodes_to_insert > 0 {
            let num_children = if nodes_to_insert >= 4 { 4 } else { nodes_to_insert };
            let mut node = QBVHNode::new();
            construct_qbvh(objects, t0, t1, bvh, &mut node, start_idx as usize, (start_idx + num_children) as usize);

            cur_node.num_children += 1;

            start_idx += num_children;
            nodes_to_insert -= num_children;
        }

        merge_bounds_from_children(cur_node, bvh);

        // what we really want to do is divide far enough down that we hit an invividual sphere level.
        // if we're <= 4 we distribute individual sphere as children
        // if we're < 16 we divide up into pad four blocks.
        // if we're >= 16 we run the normal split below
    }
    else 
    {
        sort(sub_objects, split_axis, t0, t1);

        let split_step = range / 4;

        let next_node_idx = bvh.len() as i32;
        cur_node.children[0] = next_node_idx;
        cur_node.children[1] = next_node_idx + 1;
        cur_node.children[2] = next_node_idx + 2;
        cur_node.children[3] = next_node_idx + 3;

        bvh.push(QBVHNode::new());
        bvh.push(QBVHNode::new());
        bvh.push(QBVHNode::new());
        bvh.push(QBVHNode::new());

        // TODO(); Should we try to fill up as many with new nodes with 4 children and discard the rest? 

        let mut node_0 = QBVHNode::new();
        construct_qbvh(objects, t0, t1, bvh, &mut node_0, start, start + split_step);
        bvh[cur_node.children[0] as usize] = node_0;

        let mut node_1 = QBVHNode::new();
        construct_qbvh(objects, t0, t1, bvh, &mut node_1, start + split_step, start + split_step * 2);
        bvh[cur_node.children[1] as usize] = node_1;

        let mut node_2 = QBVHNode::new();
        construct_qbvh(objects, t0, t1, bvh, &mut node_2, start + split_step * 2, start + split_step * 3);
        bvh[cur_node.children[2] as usize] = node_2;
    
        // If the range doesnt divide by 4 evenly, will pad the rest on the end. It could be better to
        // balance the slack across the first three ranges.
        let range_slack = range - ((range / 4) * 4);

        let mut node_3 = QBVHNode::new();
        construct_qbvh(objects, t0, t1, bvh, &mut node_3, start + split_step * 3, start + split_step + range_slack * 4);
        bvh[cur_node.children[2] as usize] = node_3;

        cur_node.num_children = 4;
        merge_bounds_from_children(cur_node, bvh);
    }

}

struct RGBImage {
    width: usize,
    height: usize,
    buffer: Vec<u8> 
}

impl RGBImage {
    fn new(width: usize, height: usize) -> RGBImage {
        RGBImage {width, height, buffer: vec![0u8; width * height * 3]}
    }

    fn set_pixel(&mut self, row: usize, col: usize, color: Vec3) {
        let write_idx = (col * 3) + (row * self.width * 3);
        self.buffer[write_idx] =     color.x as u8;
        self.buffer[write_idx + 1] = color.y as u8;
        self.buffer[write_idx + 2] = color.z as u8;
    }
}

fn main() {
    let aspect_ratio = 3.0 / 2.0;
    let origin = vec3![13.0, 2.0, 3.0];
    let look_at = vec3![0.0, 0.0, 0.0];
    let up = vec3![0.0, 1.0, 0.0];

    let cam = Camera::new(
        origin, look_at, up,
        20.0, aspect_ratio,
        0.1, 10.0
    );

    let pixels_x = 800_usize;
    let pixels_y = (pixels_x as f32 / aspect_ratio) as usize;
    let mut image = RGBImage::new(pixels_x, pixels_y);

    let mut objects = Vec::new();
    let mut materials = Vec::new();
    generate_world_data(&mut objects, &mut materials);
    
    let start_t = 0.0;
    let end_t = 1.0;

    let obj_count = objects.len();

    let mut bvh_tree : Vec<BVHNode> = Vec::new();
    let mut bvh_root = BVHNode::new();
    construct_bvh(&mut objects, start_t, end_t, &mut bvh_tree, &mut bvh_root, 0, obj_count);
    let bvh = BVH { root: bvh_root, tree: bvh_tree };

    // let mut qbvh_tree : Vec<QBVHNode> = Vec::new();
    // let mut qbvh_root = QBVHNode::new();
    // construct_qbvh(&mut objects, start_t, end_t, &mut qbvh_tree, &mut qbvh_root, 0, obj_count);
    // let qbvh = QBVH { root: qbvh_root, tree: qbvh_tree };

    let world = World::construct(&objects);

    println!("Finished setup. Begin rendering...");
    let measure_start = Instant::now();

    let sample_count = 8;

    for row in 0..image.height {
        for col in 0..image.width {
            let mut color = vec3![0.0, 0.0, 0.0];
            for _s in 0..sample_count {
                let (u_s, v_s) : (f32, f32) = (rand_f32(), rand_f32());
                let u = (col as f32 + u_s) / image.width as f32;
                let v = 1.0 - ((row as f32 + v_s) / image.height as f32); // Invert y to align with output from the book.
                let r = cam.get_ray(u, v, start_t, end_t);
                color += sample_color(r, &bvh, &world, &materials, 50);
            }
            
            color *= 1.0 / sample_count as f32; // Average over samples            
            color.x = 256.0 * clamp(color.x.sqrt(), 0.0, 0.999); // Gamma2 correct and move to range 0 <-> 255.
            color.y = 256.0 * clamp(color.y.sqrt(), 0.0, 0.999);
            color.z = 256.0 * clamp(color.z.sqrt(), 0.0, 0.999);
            
            image.set_pixel(row, col, color);
        }
    }

    let measure_end = Instant::now();
    let render_duration = measure_end - measure_start;

    println!("Finished rendering. Time elapsed: {} seconds.", render_duration.as_secs());
    println!("Saving result to out.png");

    let save_result = image::save_buffer("out.png", &image.buffer, image.width as u32, image.height as u32, image::ColorType::Rgb8);
    if let Err(e) = save_result {
        println!("Error saving image: {}", e);
    }
}