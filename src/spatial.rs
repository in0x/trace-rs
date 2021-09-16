use super::simd::*;
use super::math::*;
use super::collide::*;
use super::world::*;

pub trait Hitable {
    fn hit(&self, ray: Ray, t_min: f32, t_max: f32, world: &World, out_hit: &mut HitRecord) -> bool;
}

#[derive(Copy, Clone, Default)]
pub struct PackedIdx {
    value: i32
}

/// A QBVHNode has up to four children that it stores the bounds of.
/// This allows fast intersection testing by testing all 4 at once through SIMD.
struct QBVHNode {
    bb_min_x: [f32;4],
    bb_min_y: [f32;4],
    bb_min_z: [f32;4],
    bb_max_x: [f32;4],
    bb_max_y: [f32;4],
    bb_max_z: [f32;4],
    children: [PackedIdx; 4],
}

impl PackedIdx {
    fn new_joint(idx: u32) -> PackedIdx {
        PackedIdx {
            value: (idx as i32) & !i32::MIN
        }
    }

    fn joint_get_idx(&self) -> u32 {
        debug_assert!(!self.is_leaf());
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
    assert!((len % SIMD_WIDTH) == 0);
    
    let ro_x = f32x4::splat(r.origin.x);
    let ro_y = f32x4::splat(r.origin.y);
    let ro_z = f32x4::splat(r.origin.z);
    let rd_x = f32x4::splat(r.direction.x);
    let rd_y = f32x4::splat(r.direction.y);
    let rd_z = f32x4::splat(r.direction.z);
    
    let t_min4 = f32x4::splat(t_min);
    let zero_4 = f32x4::splat(0.0);

    let mut id = i32x4::splat(-1);
    let mut closest_t = f32x4::splat(t_max);

    let first_idx = start as i32;
    let mut cur_id = i32x4::set(first_idx, first_idx + 1, first_idx + 2, first_idx + 3);

    let end = start + len - (len % SIMD_WIDTH);
    for i in (start..end).step_by(SIMD_WIDTH) {    
        let sc_x = f32x4::loadu(&world.sphere_x[i..i+4]);
        let sc_y = f32x4::loadu(&world.sphere_y[i..i+4]);
        let sc_z = f32x4::loadu(&world.sphere_z[i..i+4]);

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

            id = i32x4::blendv(id, cur_id, mask);
            closest_t = f32x4::blendv(best_t, closest_t, mask);
        }
        
        cur_id = i32x4::add(cur_id, i32x4::splat(SIMD_WIDTH as i32));
    }
    
    // We found the four best hits, lets see if any are actually close.
    let min_t = f32x4::hmin(closest_t);
    if min_t < t_max { 
        let min_t_channel = f32x4::cmp_eq(closest_t, f32x4::splat(min_t));
        let min_t_mask = f32x4::movemask(min_t_channel);
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
            out_hit.normal = (out_hit.point - world.sphere_c[hit_id]) / world.sphere_r[hit_id];
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
        let center = world.sphere_c[i];
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
        let center = world.sphere_c[hit_id as usize];
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

pub struct QBVH {
    tree: Vec<QBVHNode>
}

impl QBVH {
    pub fn new() -> QBVH {
        QBVH { tree: Vec::new() }
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
    
    pub fn hit_qbvh(bvh: &QBVH, r: Ray, t_min: f32, t_max: f32, world: &World, nodes_to_check: &mut Vec<PackedIdx>, out_hit: &mut HitRecord) -> bool {
        nodes_to_check.clear();
        nodes_to_check.push(PackedIdx::new_joint(0));

        let mut hit_any_node = false;
        let mut closest_t = t_max;
        
        while nodes_to_check.len() > 0 {
            let next_node = nodes_to_check.pop().unwrap();
            
            if next_node.is_leaf() {
                let (c_idx, c_count) = next_node.leaf_get_idx_count();
                if hit_simd(world, c_idx as usize, c_count as usize, r, t_min, closest_t, out_hit) {
                    hit_any_node = true;
                    closest_t = out_hit.t;
                }
            } else {
                let bvh_node = &bvh.tree[next_node.joint_get_idx() as usize];
                let (hit_mask, any_hit) = QBVH::hit_node_children(bvh_node, r, t_min, t_max);
                if any_hit {
                    if hit_mask[0] { nodes_to_check.push(bvh_node.children[0]) }; 
                    if hit_mask[1] { nodes_to_check.push(bvh_node.children[1]) };
                    if hit_mask[2] { nodes_to_check.push(bvh_node.children[2]) };
                    if hit_mask[3] { nodes_to_check.push(bvh_node.children[3]) };
                }
            }
        }

        hit_any_node
    }
}

fn pick_split_axis(objects: &mut [Sphere], start: usize, end: usize) -> (Axis, f32) {
    let mut bounds = objects[start].calc_aabb();
    for i in start+1..end {
        bounds = AABB::merge(&bounds, &objects[i].calc_aabb())
    }

    let split_axis = bounds.longest_axis();
    let split_point = (bounds.max_at_axis(split_axis) + bounds.min_at_axis(split_axis)) * 0.5;

    (split_axis, split_point)
}

fn partition(objects: &mut [Sphere], axis: Axis, position: f32, start: usize, end: usize) -> usize {
    let mut split_idx = start;
    for i in start..end {
        let bounds = objects[i].calc_aabb();
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

    pub fn build(&mut self, objects: &mut[Sphere], start: usize, end: usize, parent: i32, child: i32, depth: u32) {
        let max_elem_in_leaf = 16;
    
        let mut bounds = objects[start].calc_aabb();
        for i in start+1..end {
            bounds = AABB::merge(&bounds, &objects[i].calc_aabb());
        }
    
        if (end - start) <= max_elem_in_leaf {
            let group_width = end - start;
            let aligned_width = ((group_width + (SIMD_WIDTH - 1)) / SIMD_WIDTH) * SIMD_WIDTH;

            let (aligned_start, aligned_end) = {
                let misalignment = aligned_width - group_width; 
                if  misalignment > 0 {
                    if (end + misalignment) < objects.len() {
                        (start, end + misalignment) 
                    } else {
                        (start - misalignment, end)
                    }
                } else {
                    (start, end)
                }
            };
            
            self.create_leaf_node(aligned_start, aligned_end, parent, child, &bounds);
            return;
        }
            
        let (split_axis, split_point) = pick_split_axis(objects, start, end);
        let split_idx = partition(objects, split_axis, split_point, start, end);

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
    
        self.build(objects, start, split_idx, current as i32, left, depth + 1);
        self.build(objects, split_idx, end, current as i32, right, depth + 1);
    }
}
