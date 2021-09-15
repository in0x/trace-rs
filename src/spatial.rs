use super::simd::*;
use super::math::*;
use super::collide::*;
use super::world::*;

pub trait Hitable {
    fn hit(&self, ray: Ray, t_min: f32, t_max: f32, world: &World, out_hit: &mut HitRecord) -> bool;
}

struct PackedIdx {
    value: i32
}

/// A QBVHNode has up to four children that it stores the bounds of.
/// This allows fast intersection testing by testing all 4 at once through SIMD.
struct QBVHNode {
    bb_min_x: [f32; 4],
    bb_min_y: [f32; 4],
    bb_min_z: [f32; 4],
    bb_max_x: [f32; 4],
    bb_max_y: [f32; 4],
    bb_max_z: [f32; 4],
    children: [PackedIdx; 4],
}

impl PackedIdx {
    fn new_joint(idx: u32) -> PackedIdx {
        PackedIdx {
            value: (idx as i32) & !i32::MIN
        }
    }

    fn joint_get_idx(&self) -> u32 {
        debug_assert_eq!(self.is_leaf(), false);
        self.value as u32
    }

    fn new_empty_leaf() -> PackedIdx {
        PackedIdx {
            value: i32::MIN
        }
    }

    const MAX_IDX: u32 = 8388607;
    const MAX_COUNT: u32 = 255; 

    /// Encodes bits as: IsChild:[1] Number of primitves:[8] start idx[23]:
    fn new_leaf(idx: u32, count: u32) -> PackedIdx {
        debug_assert!(idx <= PackedIdx::MAX_IDX, "Val: {}, Max: {}", idx, PackedIdx::MAX_IDX);
        debug_assert!(count <= PackedIdx::MAX_COUNT, "Val: {}, Max: {}", count, PackedIdx::MAX_COUNT);

        let packed_idx = ((!0 >> 9) & idx) as i32;
        let packed_count = (count << 23) as i32;

        PackedIdx {
            value: i32::MIN | packed_count | packed_idx
        }
    }

    fn is_leaf(&self) -> bool {
        (self.value & i32::MIN) == i32::MIN
    }

    fn is_empty_leaf(&self) -> bool {
        self.value == i32::MIN
    }

    fn leaf_get_idx_count(&self) -> (u32, u32) {
        debug_assert!(self.is_leaf());

        let idx = self.value as u32 & (!0 >> 9);
        let count = (self.value & !i32::MIN) >> 23;

        (idx as u32, count as u32)
    }
}

#[cfg(test)]
mod packed_idx_tests {
    use super::*;

    #[test]
    fn test_is_leaf()
    {
        let a = PackedIdx::new_leaf(4023, 15);
        assert!(a.is_leaf());
    
        let b = PackedIdx::new_empty_leaf();
        assert!(b.is_leaf());
    
        let c = PackedIdx::new_joint(u32::MAX);
        assert!(c.is_leaf() == false);
    
        let d = PackedIdx::new_joint(239);
        assert!(d.is_leaf() == false);
    }

    #[test]
    fn test_get_leaf_data() {
        {
            let idx = 4023;
            let count = 15;

            let x = PackedIdx::new_leaf(idx, count);
            let (i, c) = x.leaf_get_idx_count();
            
            assert_eq!(idx, i);
            assert_eq!(count, c);
        }
        {
            let x = PackedIdx::new_empty_leaf();
            let (i, c) = x.leaf_get_idx_count();
            assert_eq!(i, 0);
            assert_eq!(c, 0);
        }    
        {
            let x = PackedIdx::new_leaf(PackedIdx::MAX_IDX, PackedIdx::MAX_COUNT);
            let (i, c) = x.leaf_get_idx_count();
            assert_eq!(i, PackedIdx::MAX_IDX);
            assert_eq!(c, PackedIdx::MAX_COUNT);
        }
    }
}

impl QBVHNode {
    fn new() -> QBVHNode {
        QBVHNode {
            bb_min_x: [0.0, 0.0, 0.0, 0.0],
            bb_min_y: [0.0, 0.0, 0.0, 0.0],
            bb_min_z: [0.0, 0.0, 0.0, 0.0],
            bb_max_x: [0.0, 0.0, 0.0, 0.0],
            bb_max_y: [0.0, 0.0, 0.0, 0.0],
            bb_max_z: [0.0, 0.0, 0.0, 0.0],
            children: [PackedIdx::new_empty_leaf(), PackedIdx::new_empty_leaf(), PackedIdx::new_empty_leaf(), PackedIdx::new_empty_leaf()],
        }
    }

    fn set_child_joint_node(&mut self, child: usize, index: usize) {
        self.children[child] = PackedIdx::new_joint(index as u32)
    }

    fn set_child_leaf_node(&mut self, child: usize, size: usize, index: usize) {
        if size != 0 {
            self.children[child] = PackedIdx::new_leaf(index as u32, size as u32)
        }
        else {
            self.children[child] = PackedIdx::new_empty_leaf()
        }
    }

    fn set_bounds(&mut self, child: usize, bounds: &AABB) {
        self.bb_min_x[child] = bounds.min.x;
        self.bb_min_y[child] = bounds.min.y;
        self.bb_min_z[child] = bounds.min.z;
    
        self.bb_max_x[child] = bounds.max.x;
        self.bb_max_y[child] = bounds.max.y;
        self.bb_max_z[child] = bounds.max.z;    
    }
}

pub const SIMD_WIDTH: usize = 4;

fn hit_simd(world: &World, start: usize, len: usize, r: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool {
    let ro_x = f32x4::splat(r.origin.x);
    let ro_y = f32x4::splat(r.origin.y);
    let ro_z = f32x4::splat(r.origin.z);
    let rd_x = f32x4::splat(r.direction.x);
    let rd_y = f32x4::splat(r.direction.y);
    let rd_z = f32x4::splat(r.direction.z);
    let r_t  = f32x4::splat(r.time);

    let t_min4 = f32x4::splat(t_min);
    let zero_4 = f32x4::splat(0.0);

    let mut id = i32x4::splat(-1);
    let mut closest_t = f32x4::splat(t_max);

    let first_idx = start as i32;
    let mut cur_id = i32x4::set(first_idx, first_idx + 1, first_idx + 2, first_idx + 3);

    let end = start + len - (len % SIMD_WIDTH);
    for i in (start..end).step_by(SIMD_WIDTH) {
        let vel_x =  f32x4::loadu(&world.animations.velocity_x[i..i+4]);
        let vel_y =  f32x4::loadu(&world.animations.velocity_y[i..i+4]);
        let vel_z =  f32x4::loadu(&world.animations.velocity_z[i..i+4]);
        let time_0 = f32x4::loadu(&world.animations.start_time[i..i+4]);
        let time_1 = f32x4::loadu(&world.animations.end_time[i..i+4]);

        let time_t = f32x4::div(f32x4::sub(r_t, time_0), f32x4::sub(time_1, time_0));
    
        let sc_x = f32x4::add(f32x4::loadu(&world.sphere_x[i..i+4]), f32x4::mul(vel_x, time_t));
        let sc_y = f32x4::add(f32x4::loadu(&world.sphere_y[i..i+4]), f32x4::mul(vel_y, time_t));
        let sc_z = f32x4::add(f32x4::loadu(&world.sphere_z[i..i+4]), f32x4::mul(vel_z, time_t));

        let _radius =  f32x4::loadu(&world.sphere_r[i..i+4]);
        let sq_radius = f32x4::mul(_radius, _radius);

        let co_x = f32x4::sub(sc_x, ro_x);
        let co_y = f32x4::sub(sc_y, ro_y);
        let co_z = f32x4::sub(sc_z, ro_z);

        // Note: nb is minus b, avoids having to negate b (and the sign cancels out in c).
        let a  = f32x4::add(f32x4::add(f32x4::mul(rd_x, rd_x), f32x4::mul(rd_y, rd_y)), f32x4::mul(rd_z, rd_z)); 
        let nb = f32x4::add(f32x4::add(f32x4::mul(co_x, rd_x), f32x4::mul(co_y, rd_y)), f32x4::mul(co_z, rd_z));
        let c  = f32x4::sub(f32x4::add(f32x4::add(f32x4::mul(co_x, co_x), f32x4::mul(co_y, co_y)), f32x4::mul(co_z, co_z)), sq_radius);

        let discr = f32x4::sub(f32x4::mul(nb, nb), f32x4::mul(a, c));
        let discr_gt_0 = f32x4::cmp_gt(discr, zero_4);

        // See if any of the four spheres have a solution.
        if discr_gt_0.any() {
            let discr_root = f32x4::sqrt(discr);
            let t0 = f32x4::div(f32x4::sub(nb, discr_root), a);
            let t1 = f32x4::div(f32x4::add(nb, discr_root), a);

            let best_t = f32x4::blendv(t0, t1, f32x4::cmp_gt(t0, t_min4)); // If t0 is above min, take it (since it's the earlier hit), else try t1.
            let mask = u32x4::and(u32x4::and(discr_gt_0, f32x4::cmp_gt(best_t, t_min4)), f32x4::cmp_lt(best_t, closest_t));
            
            // if hit, take it
            id = i32x4::blendv(cur_id, id, mask);
            closest_t = f32x4::blendv(best_t, closest_t, mask);
        }
        
        cur_id = i32x4::add(cur_id, i32x4::splat(SIMD_WIDTH as i32));
    }
    
    // We found the four best hits, lets see if any are actually close.
    let min_t = f32x4::hmin(closest_t);
    if min_t < t_max { 
        let min_t_channel = f32x4::cmp_eq(closest_t, f32x4::splat(min_t));
        let min_t_mask = u32x4::movemask(min_t_channel);
        if min_t_mask != 0 {
            let mut id_scalar : [i32;4] = [0, 0, 0, 0];
            let mut closest_t_scalar : [f32;4] = [0.0, 0.0, 0.0, 0.0];

            i32x4::storeu(id, &mut id_scalar);
            f32x4::storeu(closest_t, &mut closest_t_scalar);

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
            out_hit.normal = (out_hit.point - world.sphere_center(hit_id, r.time)) / world.sphere_r[hit_id];
            out_hit.set_face_and_normal(r, out_hit.normal);
            out_hit.t = final_t;
            out_hit.obj_id = hit_id as usize;
            return true;
        }
    }
    
    false
}

fn hit(world: &World, start: usize, len: usize, ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool {
    let mut closest_t = t_max;
    let mut hit_id = -1;

    let end = start + len;
    for i in start..end {
        let center = world.sphere_center(i, ray.time);
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
        let center = world.sphere_center(hit_id as usize, ray.time);
        let radius = world.sphere_r[hit_id as usize];

        out_hit.t = closest_t;
        out_hit.obj_id = hit_id as usize;
        out_hit.point = ray.at(closest_t);
        let outward_normal = (out_hit.point - center) / radius;
        out_hit.set_face_and_normal(ray, outward_normal);
        true    
    } 
    else {
        false
    }
}   

fn hit_spheres(world: &World, start: usize, len: usize, ray: Ray, t_min: f32, t_max: f32, out_hit: &mut HitRecord) -> bool {
    let mut found_hit = false;
    let mut closest_t = t_max;

    if len >= SIMD_WIDTH {
        if hit_simd(world, start, len, ray, t_min, closest_t, out_hit) {
            found_hit = true;
            closest_t = out_hit.t;
        }
        
        let end = start + len;
        found_hit |= hit(world, end - (len % SIMD_WIDTH), len % SIMD_WIDTH, ray, t_min, closest_t, out_hit);
    }
    else {
        found_hit = hit(world, start, len, ray, t_min, closest_t, out_hit);
    }
    
    found_hit
}

pub struct QBVH {
    tree: Vec<QBVHNode>
}

impl QBVH {
    pub fn new() -> QBVH {
        QBVH {
            tree: Vec::new()
        }
    }
    
    // print name for natvis purposes
    // println!("{}", std::any::type_name::<PackedIdx>());

    fn hit_node(child_idx: usize, hit_any_node: &mut bool, r: Ray, t_min: f32, closest_t: &mut f32, current_node: &QBVHNode, bvh: &QBVH, world: &World, out_hit: &mut HitRecord) {
        let child = &current_node.children[child_idx];
        assert_eq!(child.is_empty_leaf(), false);

        if child.is_leaf() {
            let (c_idx, c_count) = child.leaf_get_idx_count();
            if hit(world, c_idx as usize, c_count as usize, r, t_min, *closest_t, out_hit) {
            // if hit_spheres(world, c_idx as usize, c_count as usize, r, t_min, *closest_t, out_hit) {
                *hit_any_node = true;
                *closest_t = out_hit.t;
            }
        }
        else {
            let child_node = &bvh.tree[child.joint_get_idx() as usize];
            if QBVH::hit_world(r, t_min, *closest_t, child_node, bvh, world, out_hit) {
                *hit_any_node = true;
                *closest_t = out_hit.t;
            }
        }
    }

    fn hit_world(r: Ray, t_min: f32, t_max: f32, current_node: &QBVHNode, bvh: &QBVH, world: &World, out_hit: &mut HitRecord) -> bool {        
        let (hit_mask, any_hit) = QBVH::hit_node_children(current_node, r, t_min, t_max);
        if !any_hit {
            return false;
        }

        let mut hit_any_node = false;
        let mut closest_t = t_max;

        if hit_mask[0] {
            QBVH::hit_node(0, &mut hit_any_node, r, t_min, &mut closest_t, current_node, bvh, world, out_hit);            
        }
        if hit_mask[1] {
            QBVH::hit_node(1, &mut hit_any_node, r, t_min, &mut closest_t, current_node, bvh, world, out_hit);            
        }
        if hit_mask[2] {
            QBVH::hit_node(2, &mut hit_any_node, r, t_min, &mut closest_t, current_node, bvh, world, out_hit);            
        }
        if hit_mask[3] {
            QBVH::hit_node(3, &mut hit_any_node, r, t_min, &mut closest_t, current_node, bvh, world, out_hit);            
        }

        return hit_any_node;
    }

    fn hit_node_children(node: &QBVHNode, r: Ray, t_min: f32, t_max: f32) -> ([bool;4], bool) {
        let min_x = f32x4::loadu(&node.bb_min_x);
        let min_y = f32x4::loadu(&node.bb_min_y);
        let min_z = f32x4::loadu(&node.bb_min_z);
        
        let max_x = f32x4::loadu(&node.bb_max_x);
        let max_y = f32x4::loadu(&node.bb_max_y);
        let max_z = f32x4::loadu(&node.bb_max_z);
    
        let ro_x = f32x4::splat(r.origin.x);
        let ro_y = f32x4::splat(r.origin.y);
        let ro_z = f32x4::splat(r.origin.z);
    
        let inv_rd_x = f32x4::splat(1.0 / r.direction.x);
        let inv_rd_y = f32x4::splat(1.0 / r.direction.y);
        let inv_rd_z = f32x4::splat(1.0 / r.direction.z);
    
        let zero_4 =  f32x4::splat(0.0);
        let mut t_min_4 = f32x4::splat(t_min);
        let mut t_max_4 = f32x4::splat(t_max);
        
        let any_x_miss;
        {
            let t0_x = f32x4::mul(f32x4::sub(min_x, ro_x), inv_rd_x);
            let t1_x = f32x4::mul(f32x4::sub(max_x, ro_x), inv_rd_x);
        
            let inv_rd_x_lt0 = f32x4::cmp_lt(inv_rd_x, zero_4);
            let swap_t0_x = f32x4::blendv(t1_x, t0_x, inv_rd_x_lt0);
            let swap_t1_x = f32x4::blendv(t0_x, t1_x, inv_rd_x_lt0);
        
            t_min_4 = f32x4::max(t_min_4, swap_t0_x);
            t_max_4 = f32x4::min(t_max_4, swap_t1_x);

            any_x_miss = f32x4::cmp_le(t_max_4, t_min_4);
        }

        let any_y_miss;
        {
            let t0_y = f32x4::mul(f32x4::sub(min_y, ro_y), inv_rd_y);
            let t1_y = f32x4::mul(f32x4::sub(max_y, ro_y), inv_rd_y);

            let inv_rd_y_lt0 = f32x4::cmp_lt(inv_rd_y, zero_4);
            let swap_t0_y = f32x4::blendv(t1_y, t0_y, inv_rd_y_lt0);
            let swap_t1_y = f32x4::blendv(t0_y, t1_y, inv_rd_y_lt0);
        
            t_min_4 = f32x4::max(t_min_4, swap_t0_y);
            t_max_4 = f32x4::min(t_max_4, swap_t1_y);
        
            any_y_miss = f32x4::cmp_le(t_max_4, t_min_4);
        }
        
        let any_z_miss;
        {
            let t0_z = f32x4::mul(f32x4::sub(min_z, ro_z), inv_rd_z);
            let t1_z = f32x4::mul(f32x4::sub(max_z, ro_z), inv_rd_z);

            let inv_rd_z_lt0 = f32x4::cmp_lt(inv_rd_z, zero_4);
            let swap_t0_z = f32x4::blendv(t1_z, t0_z, inv_rd_z_lt0);
            let swap_t1_z = f32x4::blendv(t0_z, t1_z, inv_rd_z_lt0);
        
            t_min_4 = f32x4::max(t_min_4, swap_t0_z);
            t_max_4 = f32x4::min(t_max_4, swap_t1_z);
        
            any_z_miss = f32x4::cmp_le(t_max_4, t_min_4);
        }
        
        let any_miss = u32x4::not(u32x4::or(u32x4::or(any_x_miss, any_y_miss), any_z_miss));

        (u32x4::get_flags(any_miss), any_miss.any())
    }
}

impl Hitable for QBVH {
    fn hit(&self, ray: Ray, t_min: f32, t_max: f32, world: &World, out_hit: &mut HitRecord) -> bool {
        return QBVH::hit_world(ray, t_min, t_max, &self.tree[0], &self, world, out_hit);
    }
}

fn pick_split_axis(objects: &mut [Sphere], start: usize, end: usize, t0: f32, t1: f32) -> (Axis, f32) {
    let mut bounds = objects[start].calc_aabb(t0, t1);
    for i in start+1..end {
        bounds = AABB::merge(&bounds, &objects[i].calc_aabb(t0, t1))
    }

    let split_axis = bounds.longest_axis();
    let split_point = (bounds.max_axis(split_axis) + bounds.min_axis(split_axis)) * 0.5;

    (split_axis, split_point)
}

fn partition(objects: &mut [Sphere], axis: Axis, position: f32, start: usize, end: usize, t0: f32, t1: f32) -> usize {
    let mut split_idx = start;
    for i in start..end {
        let bounds = objects[i].calc_aabb(t0, t1);
        let center = bounds.get_center();
        if center.at(axis as usize) <= position {
            objects.swap(i, split_idx);
            split_idx += 1;
        }
    }

    split_idx
}

impl QBVH {
    fn create_leaf_node(&mut self, start: usize, end: usize, mut parent: i32, child: i32, bounds: &AABB) {
        if parent < 0 { // root is a leaf node
            self.tree.push(QBVHNode::new());
            parent = 0;
        }

        self.tree[parent as usize].set_bounds(child as usize, bounds);
        // self.tree[parent as usize].set_child_leaf_node(child as usize, (end - start + 3) / 4, start);
        self.tree[parent as usize].set_child_leaf_node(child as usize, end - start, start);
    }

    fn create_joint_node(&mut self, parent: i32, child: i32, bounds: &AABB) -> i32 {
        let new_node_idx = self.tree.len();
        self.tree.push(QBVHNode::new());

        if parent >= 0 {
            self.tree[parent as usize].set_child_joint_node(child as usize, new_node_idx);
            self.tree[parent as usize].set_bounds(child as usize, bounds);    
        }

        new_node_idx as i32
    }

    pub fn build(&mut self, objects: &mut[Sphere], start: usize, end: usize, parent: i32, child: i32, depth: u32, t0: f32, t1: f32,) {
        let max_elem_in_leaf = 16;
    
        let mut bounds = objects[start].calc_aabb(t0, t1);
        for i in start+1..end {
            bounds = AABB::merge(&bounds, &objects[i].calc_aabb(t0, t1));
        }
    
        if (end - start) <= max_elem_in_leaf {
            self.create_leaf_node(start, end, parent, child, &bounds);
            return;
        }
            
        let (split_axis, split_point) = pick_split_axis(objects, start, end, t0, t1);
        let split_idx = partition(objects, split_axis, split_point, start, end, t0, t1);
    
        let current; let left; let right;
        if depth % 2 == 1 {
            current = parent;
            left = child;
            right = child + 1;
        }
        else {
            current = self.create_joint_node(parent, child, &bounds);
            left = 0;
            right = 2;
        }
    
        self.build(objects, start, split_idx, current as i32, left, depth + 1, t0, t1);
        self.build(objects, split_idx, end, current as i32, right, depth + 1, t0, t1);
    }
}