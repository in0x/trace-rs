use super::math::*;
use super::vec3;
use super::collide::*;

#[derive(Default)]
pub struct World {
    pub sphere_x: Vec<f32>,
    pub sphere_y: Vec<f32>,
    pub sphere_z: Vec<f32>,
    pub sphere_r: Vec<f32>,
    pub sphere_c: Vec<Vec3>,
    pub material_ids: Vec<u32>,
}

impl World {
    fn push_moving_sphere(&mut self, center: Vec3, radius: f32, material_id: u32) {
        self.sphere_x.push(center.x);
        self.sphere_y.push(center.y);
        self.sphere_z.push(center.z);
        self.sphere_r.push(radius);
        self.sphere_c.push(center);
        self.material_ids.push(material_id);
    }

    pub fn construct(objects: &[Sphere]) -> World {
        let mut world = World::default();
        for sphere in objects {
            world.push_moving_sphere(
                sphere.center,
                sphere.radius,
                sphere.material_id
            );
        }
        world
    }
}
