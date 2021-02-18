use super::math::*;
use super::vec3;
use super::collide::*;

#[derive(Default)]
pub struct AnimationData {
    pub velocity_x: Vec<f32>,
    pub velocity_y: Vec<f32>,
    pub velocity_z: Vec<f32>,
    pub start_time: Vec<f32>,
    pub end_time:   Vec<f32>,
}

impl AnimationData {
    fn get_velocity(&self, idx: usize) -> Vec3 {
        vec3![self.velocity_x[idx], self.velocity_y[idx], self.velocity_z[idx]]
    }
}

#[derive(Default)]
pub struct World {
    pub sphere_x: Vec<f32>,
    pub sphere_y: Vec<f32>,
    pub sphere_z: Vec<f32>,
    pub sphere_r: Vec<f32>,
    pub material_ids: Vec<u32>,
    pub animations: AnimationData,
}

impl World {
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

    pub fn sphere_center(&self, idx: usize, time: f32) -> Vec3 {
        // NOTE(): Could have is_static flag to shortcut calculations here. 
        let center = vec3![self.sphere_x[idx], self.sphere_y[idx], self.sphere_z[idx]];

        let velocity =  self.animations.get_velocity(idx);
        let time0 = self.animations.start_time[idx];
        let time1 = self.animations.end_time[idx];

        center + ((time - time0) / (time1 - time0)) * velocity 
    }

    pub fn construct(objects: &[Sphere]) -> World {
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
}