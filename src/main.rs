pub mod math;
use math::*;

extern crate image;
extern crate rand;
use rand::prelude::*;

#[derive(Clone, Copy)]
struct Ray {
    origin: Vec3,
    direction: Vec3
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray { origin, direction }
    }

    fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

struct HitRecord {
    t: f32,
    point: Vec3,
    normal: Vec3
}

impl HitRecord {
    fn new() -> HitRecord {
        HitRecord { t: 0.0, point: vec3![0.0, 0.0, 0.0], normal: vec3![0.0, 0.0, 0.0]}
    }
}

trait Hitable {
    fn hit(&self, ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool;
}

struct Sphere {
    center: Vec3,
    radius: f32
}

impl Hitable for Sphere {
    fn hit(&self, ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool {
        let oc = ray.origin - self.center;
        let a = dot(ray.direction, ray.direction);
        let b = dot(oc, ray.direction);
        let c = dot(oc, oc) - self.radius * self.radius;

        let discriminant = b*b - a*c; 
        if discriminant > 0.0 {
            let d_root = discriminant.sqrt();
            let t1 = (-b - d_root) / a;
            let t2 = (-b + d_root) / a;

            let (hit_t, was_hit) = match ((t1 < t_max && t1 > t_min), (t2 < t_max && t2 > t_min)) {
                (true, false) => (t1, true),
                (false, true) => (t2, true),
                (true, true) =>  (t1, true),
                _ => (0.0, false)
            };

            if was_hit {
                out_hit.t = hit_t;
                out_hit.point = ray.at(hit_t);
                out_hit.normal = (out_hit.point - self.center) / self.radius;
                return true;    
            }
        } 
        
        false            
    }
}

struct Camera {
    eye_origin: Vec3,
    lower_left: Vec3,
    x_extent: Vec3,
    y_extent: Vec3
}

impl Camera {
    fn get_ray(&self, u: f32, v: f32) -> Ray {
        Ray::new(self.eye_origin, self.lower_left + u * self.x_extent + v * self.y_extent)
    }
}

fn hit_spheres(spheres: &[Sphere], ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool {
    let mut found_hit = false;
    let mut closest_t = t_max;

    for sphere in spheres {
        if sphere.hit(ray, t_min, closest_t, out_hit) {
            found_hit = true;
            closest_t = out_hit.t;
        }
    }

    found_hit
}

fn sample_color(r: Ray, spheres: &[Sphere]) -> Vec3 {
    let mut hit_record = HitRecord::new();
    
    if hit_spheres(spheres, r, 0.0, f32::MAX, &mut hit_record) {
        0.5 * (hit_record.normal + 1.0)
    }
    else {
        let unit_dir = normalized(r.direction);
        let t = 0.5 * (unit_dir.y + 1.0);

        (1.0 - t) * vec3![1.0, 1.0, 1.0] + t * vec3![0.5, 0.7, 1.0]    
    }
}

fn main() {
    let img_width = 600;
    let img_height = 300;
    let pixel_components = 3;
    let mut img_buffer = vec![0u8; img_width * img_height * pixel_components];

    let cam = Camera { 
        eye_origin: vec3![0.0, 0.0, 0.0],
        lower_left: vec3![-2.0, -1.0, -1.0],
        x_extent: vec3![4.0, 0.0, 0.0],
        y_extent: vec3![0.0, 2.0, 0.0]  
    };

    let sphere_1 = Sphere {center: vec3![0.0, 0.0, -1.0], radius: 0.5};
    let sphere_2 = Sphere {center: vec3![0.0, -100.5, -1.0], radius: 100.0};
    let spheres = vec![sphere_1, sphere_2];

    let sample_count = 4;
    let mut rng = rand::thread_rng();

    for row in 0..img_height {
        for col in 0..img_width {
            let mut color = vec3![0.0, 0.0, 0.0];

            for s in 0..sample_count {
                let (u_s, v_s) : (f32, f32) = (rng.gen(), rng.gen());
                let u = (col as f32 + u_s) / img_width as f32;
                let v = 1.0 - ((row as f32 + v_s) / img_height as f32);
                let r = cam.get_ray(u, v);
                color += sample_color(r, &spheres);
            }
            
            color /= sample_count as f32;
            let write_idx = (col * pixel_components) + (row * img_width * pixel_components);
            img_buffer[write_idx] = (255.99 * color.x) as u8;
            img_buffer[write_idx + 1] = (255.99 * color.y) as u8;
            img_buffer[write_idx + 2] = (255.99 * color.z) as u8;
        }
    }

    let save_result = image::save_buffer("out.png", &img_buffer, img_width as u32, img_height as u32, image::ColorType::Rgb8);
    if let Err(e) = save_result {
        println!("Error saving image: {}", e);
    }
}