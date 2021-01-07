#[macro_use]
pub mod math;
use math::*;

extern crate image;

fn main() {
    let width = 200;
    let height = 100;
    let pixel_components = 3;
    let mut img_buffer = vec![0u8; width * height * pixel_components];

    for row in 0..height {
        for col in 0..width {
            let r = col as f32 / width as f32;
            let g = row as f32 / height as f32;
            let b = 0.2; 
            
            let write_idx = (col * pixel_components) + (row * width * pixel_components);
            img_buffer[write_idx] = (255.99 * r) as u8;
            img_buffer[write_idx + 1] = (255.99 * g) as u8;
            img_buffer[write_idx + 2] = (255.99 * b) as u8;            
        }
    }

    image::save_buffer("out.png", &img_buffer, width as u32, height as u32, image::ColorType::Rgb8);
}
