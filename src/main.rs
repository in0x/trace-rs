pub mod math;
use math::*;
use std::time::Instant;
use std::arch::x86_64::*;

extern crate image;
extern crate random_fast_rng;

struct Camera {
    eye_origin: Vec3,
    lower_left: Vec3,
    x_extent: Vec3,
    y_extent: Vec3,
    u: Vec3,
    v: Vec3,
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
            u, v,
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
    point: Vec3,
    normal: Vec3,
    t: f32,
    obj_id: usize,
    front_face: bool
}

impl HitRecord {
    fn new() -> HitRecord {
        HitRecord { point: vec3![0.0, 0.0, 0.0], normal: vec3![0.0, 0.0, 0.0], t: 0.0, obj_id: 0, front_face: false}
    }
    
    fn set_face_and_normal(&mut self, r: Ray, outward_normal: Vec3) {
        self.front_face = dot(r.direction, outward_normal) < 0.0;
        self.normal = if self.front_face { outward_normal } else { -outward_normal };
    }
}

trait Hitable {
    fn hit(&self, ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool;
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

fn hmin(mut m: __m128) -> f32 {
    unsafe {
        m = _mm_min_ps(m, MY_SHUFFLE_4!(m, 2, 3, 0, 0));
        m = _mm_min_ps(m, MY_SHUFFLE_4!(m, 1, 0, 0, 0));
        _mm_cvtss_f32(m)
    }
}

fn hit_simd(world: &World, r: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool {
unsafe {
    let ro_x = _mm_set_ps1(r.origin.x);
    let ro_y = _mm_set_ps1(r.origin.y);
    let ro_z = _mm_set_ps1(r.origin.z);
    let rd_x =  _mm_set_ps1(r.direction.x);
    let rd_y =  _mm_set_ps1(r.direction.y);
    let rd_z =  _mm_set_ps1(r.direction.z);

    let t_min4 = _mm_set_ps1(t_min);
    let zero_4 = _mm_set_ps1(0.0);

    let mut id = _mm_set1_epi32(-1);
    let mut closest_t = _mm_set_ps1(t_max);
    let mut cur_id = _mm_set_epi32(3, 2, 1, 0);

    let sphere_count = world.sphere_x.len() - (world.sphere_x.len() % 4);
    for i in (0..sphere_count).step_by(4) {
        let sc_x =  _mm_loadu_ps(&world.sphere_x[i]);
        let sc_y =  _mm_loadu_ps(&world.sphere_y[i]);
        let sc_z =  _mm_loadu_ps(&world.sphere_z[i]);
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
        
        cur_id = _mm_add_epi32(cur_id, _mm_set1_epi32(4));
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
            out_hit.normal = (out_hit.point - world.sphere_center(hit_id)) / world.sphere_r[hit_id];
            out_hit.set_face_and_normal(r, out_hit.normal);
            out_hit.t = final_t;
            out_hit.obj_id = hit_id as usize;
            return true;
        }
    }}
    
    return false;
}

fn hit(world: &World, idx_first: usize, ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool {
    let mut closest_t = t_max;
    let mut hit_id = -1;

    let num_spheres = world.sphere_x.len();
    for i in idx_first..num_spheres {
        let center = world.sphere_center(i);
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

            // let (hit_t, was_hit) = match ((t1 < t_max && t1 > t_min), (t2 < t_max && t2 > t_min)) {
            //     (true, false) => (t1, true),
            //     (false, true) => (t2, true),
            //     (true, true) =>  (t1, true),
            //     _ => (0.0, false)
            // };
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
        let center = world.sphere_center(hit_id as usize);
        let radius = world.sphere_r[hit_id as usize];

        out_hit.t = closest_t;
        out_hit.obj_id = hit_id as usize;
        out_hit.point = ray.at(closest_t);
        let outward_normal = (out_hit.point - center) / radius;
        out_hit.set_face_and_normal(ray, outward_normal);
        return true;    
    } 
    else {
        return false;
    }
}

fn hit_spheres(world: &World, ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool {
    let mut found_hit = false;
    let mut closest_t = t_max;

    if hit_simd(world, ray, t_min, closest_t, out_hit) {
        found_hit = true;
        closest_t = out_hit.t;
    }

    let spheres_remaining = world.sphere_x.len() - (world.sphere_x.len() % 4);
    if hit(world, spheres_remaining, ray, t_min, closest_t, out_hit) {
        found_hit = true;
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

    let mut color = vec3![1.0, 1.0, 1.0];
    let mut current_ray = r;

    for _bounce in 0..bounces {
        if hit_spheres(&world, current_ray, 0.001, f32::MAX, &mut hit) {
            hit.normal = normalized(hit.normal); // The book makes a point about unit length normals but then doesnt enforce it.
            let (did_bounce, attenuation, scattered) = world.materials[hit.obj_id].scatter(&current_ray, &hit);
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
        let should_reflect = schlick_reflectance(cos_theta, refraction_ratio) > rand_f32();
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
    sphere_x: Vec<f32>,
    sphere_y: Vec<f32>,
    sphere_z: Vec<f32>,
    sphere_r: Vec<f32>,
    materials: Vec<Material>
}

impl World {
    fn new() -> World {
        World {
            sphere_x: vec![],
            sphere_y: vec![],
            sphere_z: vec![],
            sphere_r: vec![],
            materials: vec![]
        }
    }

    fn push_sphere(&mut self, center: Vec3, radius: f32) {
        self.sphere_x.push(center.x);
        self.sphere_y.push(center.y);
        self.sphere_z.push(center.z);
        self.sphere_r.push(radius);
    }

    fn sphere_center(&self, idx: usize) -> Vec3 {
        vec3![self.sphere_x[idx], self.sphere_y[idx], self.sphere_z[idx]]
    }
}

fn generate_world() -> World {
    let mut world = World::new();

    world.push_sphere(vec3![0.0, -1000.0, 0.0], 1000.0);
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
                    world.push_sphere(center, 0.2);
                    world.materials.push(Material::Lambert(Lambert{ albedo }));
                } else if material_select < 0.96 {
                    let albedo = Vec3::random_range(0.5, 1.0);
                    let rough = rand_in_range(0.0, 0.5);
                    world.push_sphere(center, 0.2);
                    world.materials.push(Material::Metal(Metal::new(albedo, rough)));
                } else {
                    world.push_sphere(center, 0.2);
                    world.materials.push(Material::Dielectric(Dielectric{ idx_of_refraction: 1.5 }));
                }
            }
        }
    }

    world.push_sphere(vec3![0.0, 1.0, 0.0], 1.0);
    world.materials.push(Material::Dielectric(Dielectric{ idx_of_refraction: 1.5 }));

    world.push_sphere(vec3![-4.0, 1.0, 0.0], 1.0);
    world.materials.push(Material::Lambert(Lambert{ albedo: vec3![0.4, 0.2, 0.1] }));

    world.push_sphere(vec3![4.0, 1.0, 0.0], 1.0);
    world.materials.push(Material::Metal(Metal{ albedo: vec3![0.7, 0.6, 0.5], roughness: 0.0 }));

    world
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

    let pixels_x = 600_usize;
    let pixels_y = (pixels_x as f32 / aspect_ratio) as usize;
    let mut image = RGBImage::new(pixels_x, pixels_y);

    let world = generate_world();
    let sample_count = 16;

    println!("Finished setup. Begin rendering...");
    let start_time = Instant::now();

    for row in 0..image.height {
        for col in 0..image.width {
            let mut color = vec3![0.0, 0.0, 0.0];
            for _s in 0..sample_count {
                let (u_s, v_s) : (f32, f32) = (rand_f32(), rand_f32());
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