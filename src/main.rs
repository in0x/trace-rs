#[macro_use]
pub mod math;
use math::*;

extern crate image;

struct Ray {
    origin: Vec3,
    direction: Vec3
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray { origin: origin, direction: direction }
    }

    fn at(&self, t: f32) -> Vec3 {
        return self.origin + self.direction * t;
    }
}

struct Camera {
    eye_origin: Vec3,
    lower_left: Vec3,
    x_extent: Vec3,
    y_extent: Vec3
}

/// Lerp between blue and white depending on magnitude of y
fn sample_color(r: Ray) -> Vec3 {
    let unit_dir = r.direction.normalized();
    let t = 0.5 * (unit_dir.y + 1.0);

    (1.0 - t) * vec3![1.0, 1.0, 1.0] + t * vec3![0.5, 0.7, 1.0]
}

fn main() {
    let img_width = 200;
    let img_height = 100;
    let pixel_components = 3;
    let mut img_buffer = vec![0u8; img_width * img_height * pixel_components];

    let cam = Camera { 
        eye_origin: vec3![0.0, 0.0, 0.0],
        lower_left: vec3![-2.0, -1.0, -1.0],
        x_extent: vec3![4.0, 0.0, 0.0],
        y_extent: vec3![0.0, 2.0, 0.0]  
    };

    for row in 0..img_height {
        for col in 0..img_width {
            let u = col as f32 / img_width as f32;
            let v = row as f32 / img_height as f32;
            
            let r = Ray::new(cam.eye_origin, cam.lower_left + u * cam.x_extent + v * cam.y_extent);
            let color = sample_color(r);

            let write_idx = (col * pixel_components) + (row * img_width * pixel_components);
            img_buffer[write_idx] = (255.99 * color.x) as u8;
            img_buffer[write_idx + 1] = (255.99 * color.y) as u8;
            img_buffer[write_idx + 2] = (255.99 * color.z) as u8;            
        }
    }

    let save_result = image::save_buffer("out.png", &img_buffer, 
                                         img_width as u32, img_height as u32, image::ColorType::Rgb8);
    match save_result {
        Err(e) => println!("Error saving image: {}", e),
        Ok(_) => ()
    }
}