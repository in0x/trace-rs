pub mod math;
use math::*;

extern crate image;
extern crate rand;

#[derive(Clone, Copy)]
struct Ray {
    origin: Vec3,
    direction: Vec3
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray { origin, direction }
    }

    fn zero() -> Ray {
        Ray { origin: Vec3::zero(), direction: Vec3::zero() }
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

/// Returns a random point within a unit-radius sphere.
fn rand_in_unit_sphere() -> Vec3 {
    let mut p : Vec3;
    let v_ones = vec3![1.0, 1.0, 1.0]; 
    loop {
        p = 2.0 * vec3![rand::random(), rand::random(), rand::random()] - v_ones;
        if length_sq(p) < 1.0 { break; }
    }
    p
}

fn hit_spheres(spheres: &[Sphere], ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord, hit_id: &mut usize) -> bool {
    let mut found_hit = false;
    let mut closest_t = t_max;

    for (i, sphere) in spheres.iter().enumerate() {
        if sphere.hit(ray, t_min, closest_t, out_hit) {
            found_hit = true;
            closest_t = out_hit.t;
            *hit_id = i;
        }
    }

    found_hit
}

fn get_sky_color(r: Ray) -> Vec3 {
    let unit_dir = normalized(r.direction);
    let t = 0.5 * (unit_dir.y + 1.0);
    (1.0 - t) * vec3![1.0, 1.0, 1.0] + t * vec3![0.5, 0.7, 1.0]    
}

fn sample_color(r: Ray, world: &World, bounces: u32) -> Vec3 {
    let mut hit = HitRecord::new();
    let mut hit_id = 0;

    if hit_spheres(&world.objects, r, 0.001, f32::MAX, &mut hit, &mut hit_id) {
        hit.normal = normalized(hit.normal); // The book makes a point about unit length normals but then doesnt enforce it.
        if bounces < 50 {
            let (did_hit, attenuation, scattered) = world.materials[hit_id].scatter(&r, &hit);
            if did_hit {
                return attenuation * sample_color(scattered, &world, bounces + 1);
            }
        }
        Vec3::zero()
    }
    else {
        get_sky_color(r)
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

trait Material {
    /// Returns a triplet of (did_hit, attenutation, scattered_ray).
    fn scatter(&self, r: &Ray, hit: &HitRecord) -> (bool, Vec3, Ray);
}

struct Lambert {
    albedo: Vec3
}

impl Material for Lambert {
    fn scatter(&self, _r: &Ray, hit: &HitRecord) -> (bool, Vec3, Ray) {
        let target = hit.point + hit.normal + rand_in_unit_sphere();
        let scattered = Ray::new(hit.point, target - hit.point);
        let attenuation = self.albedo;
        (true, attenuation, scattered)
    }
}

struct Metal {
    albedo: Vec3,
    roughness: f32 
}

impl Material for Metal {
    fn scatter(&self, r: &Ray, hit: &HitRecord,) -> (bool, Vec3, Ray) {
        let reflected = reflect(normalized(r.direction), hit.normal);
        let scattered = Ray::new(hit.point, reflected + self.roughness * rand_in_unit_sphere());
        let attenuation = self.albedo;
        let did_hit = dot(scattered.direction, hit.normal) > 0.0;
        (did_hit, attenuation, scattered)
    }
}

enum MaterialType {
    Lambert(Lambert),
    Metal(Metal)
}

impl Material for MaterialType {
    fn scatter(&self, r: &Ray, hit: &HitRecord) -> (bool, Vec3, Ray) {
        match self {
            MaterialType::Lambert(mat) => mat.scatter(r, hit),
            MaterialType::Metal(mat) => mat.scatter(r, hit),
        }
    }
}

struct World {
    objects: Vec<Sphere>,
    materials: Vec<MaterialType>
}

fn main() {
    let mut image = RGBImage::new(600, 300);

    let cam = Camera { 
        eye_origin: vec3![0.0, 0.0, 0.0],
        lower_left: vec3![-2.0, -1.0, -1.0],
        x_extent: vec3![4.0, 0.0, 0.0],
        y_extent: vec3![0.0, 2.0, 0.0]  
    };

    let sphere_1 = Sphere {center: vec3![0.0, 0.0, -1.0], radius: 0.5};
    let sphere_2 = Sphere {center: vec3![0.0, -100.5, -1.0], radius: 100.0};
    let sphere_3 = Sphere {center: vec3![1.0, 0.0, -1.0], radius: 0.5};
    let sphere_4 = Sphere {center: vec3![-1.0, 0.0, -1.0], radius: 0.5};

    let mat_1 = Lambert { albedo: vec3![0.8, 0.3, 0.3] };
    let mat_2 = Lambert { albedo: vec3![0.8, 0.8, 0.0] };
    let mat_3 = Metal { albedo: vec3![0.8, 0.6, 0.2], roughness: 1.0 };
    let mat_4 = Metal { albedo: vec3![0.8, 0.8, 0.8], roughness: 0.3 };

    let mut world = World { objects: vec![], materials: vec![] };
    world.objects = vec![sphere_1, sphere_2, sphere_3, sphere_4];
    world.materials = vec![
        MaterialType::Lambert(mat_2), 
        MaterialType::Lambert(mat_1), 
        MaterialType::Metal(mat_3), 
        MaterialType::Metal(mat_4)];

    let sample_count = 8;

    for row in 0..image.height {
        for col in 0..image.width {
            let mut color = vec3![0.0, 0.0, 0.0];
            for _s in 0..sample_count {
                let (u_s, v_s) : (f32, f32) = (rand::random(), rand::random());
                let u = (col as f32 + u_s) / image.width as f32;
                let v = 1.0 - ((row as f32 + v_s) / image.height as f32); // Invert y to align with output from the book.
                let r = cam.get_ray(u, v);
                color += sample_color(r, &world, 0);
            }
            
            color /= sample_count as f32; // Average over samples.
            color = vec3![color.x.sqrt(), color.y.sqrt(), color.z.sqrt()]; // Gamma2 correct.
            color *= 255.99; // Move to range 0 <-> 255.
            image.set_pixel(row, col, color);
        }
    }

    let save_result = image::save_buffer("out.png", &image.buffer, image.width as u32, image.height as u32, image::ColorType::Rgb8);
    if let Err(e) = save_result {
        println!("Error saving image: {}", e);
    }
}