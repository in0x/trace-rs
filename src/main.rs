pub mod math;
pub mod camera;
pub mod collide;

use math::*;
use std::time::Instant;
use std::arch::x86_64::*;

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
    // fn push_sphere(&mut self, center: Vec3, radius: f32) {
    //     self.sphere_x.push(center.x);
    //     self.sphere_y.push(center.y);
    //     self.sphere_z.push(center.z);
    //     self.sphere_r.push(radius);
    //     self.animations.velocity_x.push(0.0);
    //     self.animations.velocity_y.push(0.0);
    //     self.animations.velocity_z.push(0.0);
    //     self.animations.start_time.push(0.0);
    //     self.animations.end_time.push(1.0);
    // }

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

    fn count(&self) -> usize {
        self.sphere_x.len()
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

unsafe fn hmin(mut m: __m128) -> f32 {
    m = _mm_min_ps(m, MY_SHUFFLE_4!(m, 2, 3, 0, 0));
    m = _mm_min_ps(m, MY_SHUFFLE_4!(m, 1, 0, 0, 0));
    _mm_cvtss_f32(m)
}

unsafe fn hmax(mut m: __m128) -> f32 {
    m = _mm_max_ps(m, MY_SHUFFLE_4!(m, 2, 3, 0, 0));
    m = _mm_max_ps(m, MY_SHUFFLE_4!(m, 1, 0, 0, 0));
    _mm_cvtss_f32(m)
}

const SIMD_WIDTH: usize = 4;

fn hit_simd(world: &World, start: usize, len: usize, r: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool {
unsafe {
    let ro_x = _mm_set_ps1(r.origin.x);
    let ro_y = _mm_set_ps1(r.origin.y);
    let ro_z = _mm_set_ps1(r.origin.z);
    let rd_x = _mm_set_ps1(r.direction.x);
    let rd_y = _mm_set_ps1(r.direction.y);
    let rd_z = _mm_set_ps1(r.direction.z);
    let r_t  = _mm_set_ps1(r.time);

    let t_min4 = _mm_set_ps1(t_min);
    let zero_4 = _mm_set_ps1(0.0);

    let mut id = _mm_set1_epi32(-1);
    let mut closest_t = _mm_set_ps1(t_max);

    let first_idx = start as i32;
    let mut cur_id = _mm_set_epi32(first_idx + 3, first_idx + 2, first_idx + 1, first_idx);

    let end = start + len;
    let sphere_count = end - (end % SIMD_WIDTH);
    for i in (start..sphere_count).step_by(SIMD_WIDTH) {
        let vel_x = _mm_loadu_ps(&world.animations.velocity_x[i]);
        let vel_y = _mm_loadu_ps(&world.animations.velocity_y[i]);
        let vel_z = _mm_loadu_ps(&world.animations.velocity_z[i]);
        let time_0 = _mm_loadu_ps(&world.animations.start_time[i]);
        let time_1 = _mm_loadu_ps(&world.animations.end_time[i]);

        let time_t = _mm_div_ps(_mm_sub_ps(r_t, time_0), _mm_sub_ps(time_1, time_0));
    
        let sc_x = _mm_add_ps(_mm_loadu_ps(&world.sphere_x[i]), _mm_mul_ps(vel_x, time_t));
        let sc_y = _mm_add_ps(_mm_loadu_ps(&world.sphere_y[i]), _mm_mul_ps(vel_y, time_t));
        let sc_z = _mm_add_ps(_mm_loadu_ps(&world.sphere_z[i]), _mm_mul_ps(vel_z, time_t));

        let _radius =  _mm_loadu_ps(&world.sphere_r[i]);
        let sq_radius = _mm_mul_ps(_radius, _radius);

        let co_x = _mm_sub_ps(sc_x, ro_x);
        let co_y = _mm_sub_ps(sc_y, ro_y);
        let co_z = _mm_sub_ps(sc_z, ro_z);

        // Note: nb is minus b, avoids having to negate b (and the sign cancels out in c).
        let a  = _mm_add_ps(_mm_add_ps(_mm_mul_ps(rd_x, rd_x), _mm_mul_ps(rd_y, rd_y)), _mm_mul_ps(rd_z, rd_z)); 
        let nb = _mm_add_ps(_mm_add_ps(_mm_mul_ps(co_x, rd_x), _mm_mul_ps(co_y, rd_y)), _mm_mul_ps(co_z, rd_z));
        let c  = _mm_sub_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(co_x, co_x), _mm_mul_ps(co_y, co_y)), _mm_mul_ps(co_z, co_z)), sq_radius);

        let discr = _mm_sub_ps(_mm_mul_ps(nb, nb), _mm_mul_ps(a, c));
        let discr_gt_0 = _mm_cmpgt_ps(discr, zero_4);

        // See if any of the four spheres have a solution.
        if  _mm_movemask_ps(discr_gt_0) != 0 {
            let discr_root = _mm_sqrt_ps(discr);
            let t0 = _mm_div_ps(_mm_sub_ps(nb, discr_root), a);
            let t1 = _mm_div_ps(_mm_add_ps(nb, discr_root), a);

            let best_t = select(t1, t0, _mm_cmpgt_ps(t0, t_min4)); // If t0 is above min, take it (since it's the earlier hit), else try t1.
            let mask = _mm_and_ps(_mm_and_ps(discr_gt_0, _mm_cmpgt_ps(best_t, t_min4)), _mm_cmplt_ps(best_t, closest_t));
            
            // if hit, take it
            id = selecti(id, cur_id, mask);
            closest_t = select(closest_t, best_t, mask);
        }
        
        cur_id = _mm_add_epi32(cur_id, _mm_set1_epi32(SIMD_WIDTH as i32));
    }
    
    // We found the four best hits, lets see if any are actually close.
    let min_t = hmin(closest_t);
    if min_t < t_max { 
        let min_t_channel = _mm_cmpeq_ps(closest_t, _mm_set_ps1(min_t));
        let min_t_mask = _mm_movemask_ps(min_t_channel);
        if min_t_mask != 0 {
            let mut id_scalar : [i32;4] = [0, 0, 0, 0];
            let mut closest_t_scalar : [f32;4] = [0.0, 0.0, 0.0, 0.0];

            _mm_storeu_si128(id_scalar.as_mut_ptr() as *mut __m128i, id);
            _mm_storeu_ps(closest_t_scalar.as_mut_ptr(), closest_t);

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
    }}
    
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
    fn hit(&self, ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool;
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

    let l_hit = hit_world(r, t_min, t_max, l_node, bvh, world, out_hit);
    let closest_t = if l_hit { out_hit.t } else { t_max }; 
    let r_hit = hit_world(r, t_min, closest_t, r_node, bvh, world, out_hit);

    l_hit || r_hit
}

fn sample_color(r: Ray, bvh: &BVH, world: &World, materials: &[Material], bounces: i32) -> Vec3 {
    let mut hit = HitRecord::new();

    let mut color = vec3![1.0, 1.0, 1.0];
    let mut current_ray = r;

    for _bounce in 0..bounces {
        if hit_world(current_ray, 0.001, f32::MAX, &bvh.root, bvh, &world, &mut hit) {
        // if hit_spheres(world, 0, world.sphere_x.len(), current_ray, 0.001, f32::MAX, &mut hit) {
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

struct Sphere {
    center: Vec3,
    velocity: Vec3,
    start_time: f32,
    end_time: f32,
    radius: f32,
    material_id: u32
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

    fn _hit_simd(&self, r: Ray, t_min: f32, t_max: f32) -> bool {
    unsafe {
            let inv_rd = _mm_div_ps(_mm_set_ps1(1.0), _mm_set_ps(r.direction.x, r.direction.y, r.direction.z, 1.0));
            let ro = _mm_set_ps(r.origin.x, r.origin.y, r.origin.z, 0.0);
    
            let bb_min = _mm_set_ps(self.min.x, self.min.y, self.min.z, 0.0);
            let bb_max = _mm_set_ps(self.max.x, self.max.y, self.max.z, 0.0);
    
            let t0 = _mm_mul_ps(_mm_sub_ps(bb_min, ro), inv_rd);
            let t1 = _mm_mul_ps(_mm_sub_ps(bb_max, ro), inv_rd);
    
            let inv_lt_0 = _mm_cmplt_ps(inv_rd, _mm_set_ps1(0.0));

            let mut swap_min = _mm_blendv_ps(t0, t1, inv_lt_0);
            let mut swap_max = _mm_blendv_ps(t1, t0, inv_lt_0);

            swap_min = _mm_blend_ps(swap_min, _mm_set_ps1(t_min), 1);
            swap_max = _mm_blend_ps(swap_max, _mm_set_ps1(t_max), 1);

            hmin(swap_max) > hmax(swap_min)
    }
    }
}

impl Sphere {
    fn calc_static_aabb(&self) -> AABB {
        AABB {
            min: self.center - self.radius,
            max: self.center + self.radius
        }
    }

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

fn decompose_world(objects: &[Sphere]) -> World {
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

    let mut bvh_tree : Vec<BVHNode> = Vec::new();
    let mut bvh_root = BVHNode::new();
    let obj_count = objects.len();
    construct_bvh(&mut objects, start_t, end_t, &mut bvh_tree, &mut bvh_root, 0, obj_count);

    let bvh = BVH { root: bvh_root, tree: bvh_tree };
    let world = decompose_world(&objects);

    println!("Finished setup. Begin rendering...");
    let measure_start = Instant::now();

    let sample_count = 64;

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