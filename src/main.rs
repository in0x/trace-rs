pub mod math;
use math::*;
use std::time::Instant;

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
    normal: Vec3,
    front_face: bool
}

impl HitRecord {
    fn new() -> HitRecord {
        HitRecord { t: 0.0, point: vec3![0.0, 0.0, 0.0], normal: vec3![0.0, 0.0, 0.0], front_face: false}
    }
    
    fn set_face_and_normal(&mut self, r: Ray, outward_normal: Vec3) {
        self.front_face = dot(r.direction, outward_normal) < 0.0;
        self.normal = if self.front_face { outward_normal } else { -outward_normal };
    }
}

trait Hitable {
    fn hit(&self, ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool;
}

struct Sphere {
    center: Vec3,
    radius: f32
}

impl Sphere {
    fn new(center: Vec3, radius: f32) -> Sphere {
        Sphere { center, radius }
    }
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
                let outward_normal = (out_hit.point - self.center) / self.radius;
                out_hit.set_face_and_normal(ray, outward_normal);
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
    y_extent: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f32
}

impl Camera {
    fn new(origin: Vec3, look_at: Vec3, up: Vec3, y_fov_deg: f32, aspect_ratio: f32, aperture: f32, focus_dist: f32) -> Camera {
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
            u: u, v: v, w: w,
            lens_radius: aperture / 2.0
        }
    }

    fn get_ray(&self, s: f32, t: f32) -> Ray {
        let rd = self.lens_radius * rand_in_unit_disk();
        let offset = self.u * rd.x + self.v * rd.y;

        Ray::new(
            self.eye_origin + offset, 
            self.lower_left + s * self.x_extent + t * self.y_extent - self.eye_origin - offset)
    }
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

fn sample_color(r: Ray, world: &World, bounces: i32) -> Vec3 {
    let mut hit = HitRecord::new();
    let mut hit_id = 0;

    let mut color = vec3![1.0, 1.0, 1.0];
    let mut current_ray = r;

    for _bounce in 0..bounces {
        if hit_spheres(&world.objects, current_ray, 0.001, f32::MAX, &mut hit, &mut hit_id) {
            hit.normal = normalized(hit.normal); // The book makes a point about unit length normals but then doesnt enforce it.
            let (did_bounce, attenuation, scattered) = world.materials[hit_id].scatter(&current_ray, &hit);
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

trait Scattering {
    /// Returns a triplet of (did_hit, attenutation, scattered_ray).
    fn scatter(&self, r: &Ray, hit: &HitRecord) -> (bool, Vec3, Ray);
}

struct Lambert {
    albedo: Vec3
}

impl Scattering for Lambert {
    fn scatter(&self, _r: &Ray, hit: &HitRecord) -> (bool, Vec3, Ray) {
        let mut scatter_dir = hit.normal + rand_unit_vector();
        if scatter_dir.is_near_zero() {
            scatter_dir = hit.normal;
        }

        let scattered = Ray::new(hit.point, scatter_dir);
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
        let scattered = Ray::new(hit.point, reflected + self.roughness * rand_in_unit_sphere());
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
        let should_reflect = schlick_reflectance(cos_theta, refraction_ratio) > rand::random();
        let direction = match cannot_refract || should_reflect {
            true => reflect(unit_dir, hit.normal),
            false => refract(unit_dir, hit.normal, refraction_ratio)
        };

        (true, vec3![1.0], Ray::new(hit.point, direction))
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

struct World {
    objects: Vec<Sphere>,
    materials: Vec<Material>
}

fn generate_world() -> World {
    let mut world = World { objects: vec![], materials: vec![] };

    world.objects.push(Sphere::new(vec3![0.0, -1000.0, 0.0], 1000.0));
    world.materials.push(Material::Lambert(Lambert{ albedo: vec3![0.5, 0.5, 0.5] }));

    for a in -11..11 {
        for b in -11..11 {
            let material_select : f32 = rand::random();
            let rand_a : f32 = rand::random();
            let rand_b : f32 = rand::random();
            let center = vec3![(a as f32) + 0.9 * rand_a, 0.2, (b as f32) + 0.9 * rand_b];
            
            if length(center - vec3![4.0, 0.2, 0.0]) > 0.9 {
                if material_select < 0.8 {
                    let albedo = Vec3::random() * Vec3::random();
                    world.objects.push(Sphere::new(center, 0.2));
                    world.materials.push(Material::Lambert(Lambert{ albedo: albedo }));
                } else if material_select < 0.96 {
                    let albedo = Vec3::random_range(0.5, 1.0);
                    let rough = rand_in_range(0.0, 0.5);
                    world.objects.push(Sphere::new(center, 0.2));
                    world.materials.push(Material::Metal(Metal {albedo: albedo, roughness: rough}));
                } else {
                    world.objects.push(Sphere::new(center, 0.2));
                    world.materials.push(Material::Dielectric(Dielectric{ idx_of_refraction: 1.5 }));
                }
            }
        }
    }

    world.objects.push(Sphere::new(vec3![0.0, 1.0, 0.0], 1.0));
    world.materials.push(Material::Dielectric(Dielectric{ idx_of_refraction: 1.5 }));

    world.objects.push(Sphere::new(vec3![-4.0, 1.0, 0.0], 1.0));
    world.materials.push(Material::Lambert(Lambert{ albedo: vec3![0.4, 0.2, 0.1] }));

    world.objects.push(Sphere::new(vec3![4.0, 1.0, 0.0], 1.0));
    world.materials.push(Material::Metal(Metal{ albedo: vec3![0.7, 0.6, 0.5], roughness: 0.0 }));

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

    let pixels_x = 600_usize;
    let pixels_y = (pixels_x as f32 / aspect_ratio) as usize;
    let mut image = RGBImage::new(pixels_x, pixels_y);

    let world = generate_world();
    let sample_count = 8;

    println!("Finished setup. Begin rendering...");
    let start_time = Instant::now();

    for row in 0..image.height {
        for col in 0..image.width {
            let mut color = vec3![0.0, 0.0, 0.0];
            for _s in 0..sample_count {
                let (u_s, v_s) : (f32, f32) = (rand::random(), rand::random());
                let u = (col as f32 + u_s) / image.width as f32;
                let v = 1.0 - ((row as f32 + v_s) / image.height as f32); // Invert y to align with output from the book.
                let r = cam.get_ray(u, v);
                color += sample_color(r, &world, 50);
            }
            
            color *= 1.0 / sample_count as f32; // Average over samples            
            color.x = 256.0 * clamp(color.x.sqrt(), 0.0, 0.999); // Gamma2 correct and move to range 0 <-> 255.
            color.y = 256.0 * clamp(color.y.sqrt(), 0.0, 0.999);
            color.z = 256.0 * clamp(color.z.sqrt(), 0.0, 0.999);
            
            image.set_pixel(row, col, color);
        }
    }

    let end_time = Instant::now();
    let render_duration = end_time - start_time;

    println!("Finished rendering. Time elapsed: {} seconds.", render_duration.as_secs());
    println!("Saving result to out.png");

    let save_result = image::save_buffer("out.png", &image.buffer, image.width as u32, image.height as u32, image::ColorType::Rgb8);
    if let Err(e) = save_result {
        println!("Error saving image: {}", e);
    }
}