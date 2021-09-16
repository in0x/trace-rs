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
    pub fn construct(objects: &[Sphere]) -> World {
        let mut world = World::default();
        for sphere in objects {
            world.sphere_x.push(sphere.center.x);
            world.sphere_y.push(sphere.center.y);
            world.sphere_z.push(sphere.center.z);
            world.sphere_r.push(sphere.radius);
            world.sphere_c.push(sphere.center);
            world.material_ids.push(sphere.material_id);
        }
        
        world
    }
}
