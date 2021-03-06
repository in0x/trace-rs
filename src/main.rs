#![feature(stdsimd)]
#![feature(platform_intrinsics)]
#![feature(link_llvm_intrinsics)]
#![feature(simd_ffi)]

pub mod math;
pub mod simd;
pub mod spatial;
pub mod collide;
pub mod world;

use math::*;
use simd::*;
use collide::*;
use spatial::*;
use world::*;
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

// unsafe fn hmax(mut m: __m128) -> f32 {
//     m = _mm_max_ps(m, MY_SHUFFLE_4!(m, 2, 3, 0, 0));
//     m = _mm_max_ps(m, MY_SHUFFLE_4!(m, 1, 0, 0, 0));
//     _mm_cvtss_f32(m)
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
    let end_t = 0.0;

    let mut bvh = QBVH::new();
    let obj_count = objects.len();
    bvh.build(&mut objects, 0, obj_count, -1, 0, 0, start_t, end_t);

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