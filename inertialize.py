import numpy as np
import quat
from spring import *

def inertialize_pose_reset(
        bone_offset_positions,
        bone_offset_velocities,
        bone_offset_rotations,
        bone_offset_angular_velocities,
        transition_src_position,
        transition_src_rotation,
        transition_dst_position,
        transition_dst_rotation,
        anim_root_position,
        anim_root_rotation
):
    bone_offset_positions[:] = np.array([0.0, 0.0, 0.0], dtype = np.float32)
    bone_offset_velocities[:] = np.array([0.0, 0.0, 0.0], dtype = np.float32)
    bone_offset_rotations[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype = np.float32)
    bone_offset_angular_velocities[:] = np.array([0.0, 0.0, 0.0], dtype = np.float32)

    transition_src_position[:] = anim_root_position
    transition_src_rotation[:] = anim_root_rotation
    transition_dst_position[:] = np.array([0.0, 0.0, 0.0], dtype = np.float32)
    transition_dst_rotation[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype = np.float32)

def inertialize_pose_transition(
    bone_offset_positions,
    bone_offset_velocities,
    bone_offset_rotations,
    bone_offset_angular_velocities,
    transition_src_position,
    transition_src_rotation,
    transition_dst_position,
    transition_dst_rotation,
    root_position,
    root_velocity,
    root_rotation,
    root_angular_velocity,
    bone_src_positions,
    bone_src_velocities,
    bone_src_rotations,
    bone_src_angular_velocities,
    bone_dst_positions,
    bone_dst_velocities,
    bone_dst_rotations,
    bone_dst_angular_velocities
):
    transition_dst_position[:] = root_position
    transition_dst_rotation[:] = root_rotation
    transition_src_position[:] = bone_dst_positions[0]
    transition_src_rotation[:] = bone_dst_rotations[0]

    world_space_dst_velocity = quat.mul_vec(transition_dst_rotation, quat.inv_mul_vec(transition_src_rotation, bone_dst_velocities[0]))
    world_space_dst_angular_velocity = quat.mul_vec(transition_dst_rotation, quat.inv_mul_vec(transition_src_rotation, bone_dst_angular_velocities[0]))

    bone_offset_positions[0, :] = np.array([0.0, 0.0, 0.0], dtype = np.float32)
    bone_offset_velocities[0, :] = np.array([0.0, 0.0, 0.0], dtype = np.float32)
    bone_offset_rotations[0, :] = np.array([1.0, 0.0, 0.0, 0.0], dtype = np.float32)
    bone_offset_angular_velocities[0, :] = np.array([0.0, 0.0, 0.0], dtype = np.float32)
    
    inertialize_transition(bone_offset_positions[0], bone_offset_velocities[0], root_position, root_velocity, root_position, world_space_dst_velocity)
    inertialize_transition_quat(bone_offset_rotations[0], bone_offset_angular_velocities[0], root_rotation, root_angular_velocity, root_rotation, world_space_dst_angular_velocity)

    inertialize_transition(bone_offset_positions[1:], bone_offset_velocities[1:], bone_src_positions[1:], bone_src_velocities[1:], bone_dst_positions[1:], bone_dst_velocities[1:])
    inertialize_transition_quat(bone_offset_rotations[1:], bone_offset_angular_velocities[1:], bone_src_rotations[1:], bone_src_angular_velocities[1:], bone_dst_rotations[1:], bone_dst_angular_velocities[1:])

def inertialize_pose_update(
    bone_positions,
    bone_velocities,
    bone_rotations,
    bone_angular_velocities,
    bone_offset_positions,
    bone_offset_velocities,
    bone_offset_rotations,
    bone_offset_angular_velocities,
    bone_input_positions,
    bone_input_velocities,
    bone_input_rotations,
    bone_input_angular_velocities,
    transition_src_position,
    transition_src_rotation,
    transition_dst_position,
    transition_dst_rotation,
    halflife,
    dt
):
    world_space_position = quat.mul_vec(transition_dst_rotation, quat.inv_mul_vec(transition_src_rotation, bone_input_positions[0] - transition_src_position)) + transition_dst_position
    world_space_velocity = quat.mul_vec(transition_dst_rotation, quat.inv_mul_vec(transition_src_rotation, bone_input_velocities[0]))
    world_space_rotation = quat.normalize(quat.mul(transition_dst_rotation, quat.inv_mul(transition_src_rotation, bone_input_rotations[0])))
    world_space_angular_velocity = quat.mul_vec(transition_dst_rotation, quat.inv_mul_vec(transition_src_rotation, bone_input_angular_velocities[0]))

    inertialize_update(
        bone_positions[0],
        bone_velocities[0],
        bone_offset_positions[0],
        bone_offset_velocities[0],
        world_space_position,
        world_space_velocity,
        halflife,
        dt
    )

    inertialize_update_quat(
        bone_rotations[0],
        bone_angular_velocities[0],
        bone_offset_rotations[0],
        bone_offset_angular_velocities[0],
        world_space_rotation,
        world_space_angular_velocity,
        halflife,
        dt
    )

    inertialize_update(
        bone_positions[1:],
        bone_velocities[1:],
        bone_offset_positions[1:],
        bone_offset_velocities[1:],
        bone_input_positions[1:],
        bone_input_velocities[1:],
        halflife,
        dt
    )

    inertialize_update_quat(
        bone_rotations[1:],
        bone_angular_velocities[1:],
        bone_offset_rotations[1:],
        bone_offset_angular_velocities[1:],
        bone_input_rotations[1:],
        bone_input_angular_velocities[1:],
        halflife,
        dt
    )
