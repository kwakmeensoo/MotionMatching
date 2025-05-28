import numpy as np
import moderngl as mgl
import glm
import glfw
from camera import Camera, CameraMode
from geometry import *
from objects import *
from lbs import load_character

from database import *
from spring import *
from inertialize import *

class Engine:
    def __init__(self, renderer):
        self.renderer = renderer

        self.window = self.renderer.window
        self.camera = self.renderer.camera
        self.ctx = self.renderer.ctx
        
        self._init_shaders()

        self.checkboard = CheckerboardPlane(
            self.ctx, self.programs['checkerboard'], width = 100.0, depth = 100.0
        )

        self.skeleton_color = (0.2, 0.2, 0.9)
        self.character_color = (0.8, 0.8, 0.8)

        self.skeletons = None
        self.character = None
        self.simulation_trajectory_objects = None

        self._setup()

    def _setup(self):
        self.db = Database('./resources/database.bin')
        
        ## Calculate Features
        self.feature_weight_foot_position = 0.75
        self.feature_weight_foot_velocity = 1.0
        self.feature_weight_hip_velocity = 1.0
        self.feature_weight_trajectory_positions = 1.0
        self.feature_weight_trajectory_directions = 1.5
        build_matching_features(self.db,
                                self.feature_weight_foot_position,
                                self.feature_weight_foot_velocity,
                                self.feature_weight_hip_velocity,
                                self.feature_weight_trajectory_positions,
                                self.feature_weight_trajectory_directions)
        # Frame Index
        self.frame_index = self.db.range_starts[0]

        # Pose & Inertialize Data
        self.inertialize_blending_halflife = 0.1

        self.curr_bone_positions = self.db.bone_positions[self.frame_index, :]
        self.curr_bone_velocities = self.db.bone_velocities[self.frame_index, :]
        self.curr_bone_rotations = self.db.bone_rotations[self.frame_index, :]
        self.curr_bone_angular_velocities = self.db.bone_angular_velocities[self.frame_index, :]
        
        self.trns_bone_positions = self.db.bone_positions[self.frame_index, :]
        self.trns_bone_velocities = self.db.bone_velocities[self.frame_index, :]
        self.trns_bone_rotations = self.db.bone_rotations[self.frame_index, :]
        self.trns_bone_angular_velocities = self.db.bone_angular_velocities[self.frame_index, :]
        
        self.bone_positions = np.zeros_like(self.curr_bone_positions)
        self.bone_velocities = np.zeros_like(self.curr_bone_velocities)
        self.bone_rotations = np.zeros_like(self.curr_bone_rotations)
        self.bone_angular_velocities = np.zeros_like(self.curr_bone_angular_velocities)

        self.bone_offset_positions = np.zeros_like(self.bone_positions)
        self.bone_offset_velocities = np.zeros_like(self.bone_velocities)
        self.bone_offset_rotations = np.zeros_like(self.bone_rotations)
        self.bone_offset_angular_velocities = np.zeros_like(self.bone_angular_velocities)

        self.global_bone_positions = np.zeros_like(self.curr_bone_positions)
        self.global_bone_rotations = np.zeros_like(self.curr_bone_rotations)

        self.transition_src_position = np.array([0.0, 0.0, 0.0], dtype = np.float32)
        self.transition_src_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype = np.float32)
        self.transition_dst_position = np.array([0.0, 0.0, 0.0], dtype = np.float32)
        self.transition_dst_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype = np.float32)

        inertialize_pose_reset(
            self.bone_offset_positions,
            self.bone_offset_velocities,
            self.bone_offset_rotations,
            self.bone_offset_angular_velocities,
            self.transition_src_position,
            self.transition_src_rotation,
            self.transition_dst_position,
            self.transition_dst_rotation,
            self.curr_bone_positions[0],
            self.curr_bone_rotations[0]
        )

        inertialize_pose_update(
            self.bone_positions,
            self.bone_velocities,
            self.bone_rotations,
            self.bone_angular_velocities,
            self.bone_offset_positions,
            self.bone_offset_velocities,
            self.bone_offset_rotations,
            self.bone_offset_angular_velocities,
            self.db.bone_positions[self.frame_index],
            self.db.bone_velocities[self.frame_index],
            self.db.bone_rotations[self.frame_index],
            self.db.bone_angular_velocities[self.frame_index],
            self.transition_src_position,
            self.transition_src_rotation,
            self.transition_dst_position,
            self.transition_dst_rotation,
            self.inertialize_blending_halflife,
            0.0
        )

        self.search_time = 0.1
        self.search_timer = self.search_time
        self.force_search_timer = self.search_time

        self.desired_velocity = np.zeros((3,), dtype = np.float32)
        self.desired_velocity_change_curr = np.zeros((3,), dtype = np.float32)
        self.desired_velocity_change_prev = np.zeros((3,), dtype = np.float32)
        self.desired_velocity_change_threshold = 50.0

        self.desired_rotation = np.zeros((4,), dtype = np.float32)
        self.desired_rotation_change_curr = np.zeros((3,), dtype = np.float32)
        self.desired_rotation_change_prev = np.zeros((3,), dtype = np.float32)
        self.desired_rotation_change_threshold = 50.0

        self.desired_gait = np.array(0.0, np.float32)
        self.desired_gait_velocity = np.array(0.0, np.float32)

        self.simulation_position = np.zeros((3,), dtype = np.float32)
        self.simulation_velocity = np.zeros((3,), dtype = np.float32)
        self.simulation_acceleration = np.zeros((3,), dtype = np.float32)
        self.simulation_rotation = np.zeros((4,), dtype = np.float32)
        self.simulation_angular_velocity = np.zeros((3,), dtype = np.float32)

        self.simulation_velocity_halflife = 0.27
        self.simulation_rotation_halflife = 0.27

        self.simulation_run_fwrd_speed = 4.0
        self.simulation_run_side_speed = 3.0
        self.simulation_run_back_speed = 2.5

        self.simulation_walk_fwrd_speed = 1.75
        self.simulation_walk_side_speed = 1.5
        self.simulation_walk_back_speed = 1.25

        self.trajectory_desired_velocities = np.zeros((4, 3), dtype = np.float32)
        self.trajectory_desired_rotations = np.zeros((4, 4), dtype = np.float32)
        self.trajectory_positions = np.zeros((4, 3), dtype = np.float32)
        self.trajectory_velocities = np.zeros((4, 3), dtype = np.float32)
        self.trajectory_accelerations = np.zeros((4, 3), dtype = np.float32)
        self.trajectory_rotations = np.zeros((4, 4), dtype = np.float32)
        self.trajectory_angular_velocities = np.zeros((4, 3), dtype = np.float32)


        self.global_bone_positions, self.global_bone_rotations = forward_kinematics_full(
            self.bone_positions,
            self.bone_rotations,
            self.db.bone_parents
        )

        self._init_character('./resources/character.bin')
        # self._init_skeletons()
        self._init_simulation_trajectory_objects()

    def update(self, dt):
        movement_input = self.renderer.controller.get_movement_input()
        
        self.desired_gait_update(
            self.desired_gait,
            self.desired_gait_velocity,
            dt
        )

        simulation_fwrd_speed = lerp(self.simulation_run_fwrd_speed, self.simulation_walk_fwrd_speed, self.desired_gait)
        simulation_side_speed = lerp(self.simulation_run_side_speed, self.simulation_walk_side_speed, self.desired_gait)
        simulation_back_speed = lerp(self.simulation_run_back_speed, self.simulation_walk_back_speed, self.desired_gait)

        desired_velocity_curr = self.desired_velocity_update(
            movement_input,
            self.simulation_rotation,
            simulation_fwrd_speed,
            simulation_side_speed,
            simulation_back_speed
        )

        desired_rotation_curr = self.desired_rotation_update(
            self.desired_rotation,
            movement_input,
            desired_velocity_curr
        )

        self.desired_velocity_change_prev[:] = self.desired_velocity_change_curr
        self.desired_velocity_change_curr[:] = (desired_velocity_curr - self.desired_velocity) / dt
        self.desired_velocity[:] = desired_velocity_curr

        self.desired_rotation_change_prev[:] = self.desired_rotation_change_curr
        self.desired_rotation_change_curr[:] = quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(desired_rotation_curr, self.desired_rotation))) / dt
        self.desired_rotation[:] = desired_rotation_curr

        force_search = False

        if self.force_search_timer <= 0 and ((length(self.desired_velocity_change_prev) >= self.desired_velocity_change_threshold and 
                                              length(self.desired_velocity_change_curr) < self.desired_velocity_change_threshold) or
                                             (length(self.desired_rotation_change_prev) >= self.desired_rotation_change_threshold and
                                              length(self.desired_rotation_change_curr) < self.desired_rotation_change_threshold)):
            force_search = True
            self.force_search_timer = self.search_time
        elif self.force_search_timer > 0:
            self.force_search_timer -= dt
        
        self.trajectory_desired_rotations_predict(
            self.trajectory_desired_rotations,
            self.trajectory_desired_velocities,
            self.desired_rotation,
            movement_input
        )

        self.trajectory_rotations_predict(
            self.trajectory_rotations,
            self.trajectory_angular_velocities,
            self.simulation_rotation,
            self.simulation_angular_velocity,
            self.trajectory_desired_rotations,
            self.simulation_rotation_halflife,
            20.0 * dt
        )

        self.trajectory_desired_velocities_predict(
            self.trajectory_desired_velocities,
            self.trajectory_rotations,
            self.desired_velocity,
            movement_input,
            simulation_fwrd_speed,
            simulation_side_speed,
            simulation_back_speed,
            20.0 * dt
        )

        self.trajectory_positions_predict(
            self.trajectory_positions,
            self.trajectory_velocities,
            self.trajectory_accelerations,
            self.simulation_position,
            self.simulation_velocity,
            self.simulation_acceleration,
            self.trajectory_desired_velocities,
            self.simulation_velocity_halflife,
            20.0 * dt
        )

        query = np.zeros_like(self.db.features[self.frame_index])
        self.query_copy_denormalized_feature(query, self.db, self.db.features[self.frame_index], 0, 15)
        self.query_compute_trajectory_position_feature(query, 15, self.bone_positions[0], self.bone_rotations[0], self.trajectory_positions)
        self.query_compute_trajectory_direction_feature(query, 21, self.bone_rotations[0], self.trajectory_rotations)

        end_of_anim = self.db.clamp_trajectory_index(self.frame_index, 1) == self.frame_index

        if force_search or self.search_timer or end_of_anim:
            best_index, best_cost = search(self.db, query, self.frame_index)

            if best_index != self.frame_index:
                self.trns_bone_positions = self.db.bone_positions[best_index]
                self.trns_bone_velocities = self.db.bone_velocities[best_index]
                self.trns_bone_rotations = self.db.bone_rotations[best_index]
                self.trns_bone_angular_velocities = self.db.bone_angular_velocities[best_index]

                inertialize_pose_transition(
                    self.bone_offset_positions,
                    self.bone_offset_velocities,
                    self.bone_offset_rotations,
                    self.bone_offset_angular_velocities,
                    self.transition_src_position,
                    self.transition_src_rotation,
                    self.transition_dst_position,
                    self.transition_dst_rotation,
                    self.bone_positions[0],
                    self.bone_velocities[0],
                    self.bone_rotations[0],
                    self.bone_angular_velocities[0],
                    self.curr_bone_positions,
                    self.curr_bone_velocities,
                    self.curr_bone_rotations,
                    self.curr_bone_angular_velocities,
                    self.trns_bone_positions,
                    self.trns_bone_velocities,
                    self.trns_bone_rotations,
                    self.trns_bone_angular_velocities
                )

                self.frame_index = best_index

            self.search_timer = self.search_time

        self.search_timer -= dt

        self.frame_index = self.frame_index + 1

        self.curr_bone_positions = self.db.bone_positions[self.frame_index]
        self.curr_bone_velocities = self.db.bone_velocities[self.frame_index]
        self.curr_bone_rotations = self.db.bone_rotations[self.frame_index]
        self.curr_bone_angular_velocities = self.db.bone_angular_velocities[self.frame_index]

        inertialize_pose_update(
            self.bone_positions,
            self.bone_velocities,
            self.bone_rotations,
            self.bone_angular_velocities,
            self.bone_offset_positions,
            self.bone_offset_velocities,
            self.bone_offset_rotations,
            self.bone_offset_angular_velocities,
            self.curr_bone_positions,
            self.curr_bone_velocities,
            self.curr_bone_rotations,
            self.curr_bone_angular_velocities,
            self.transition_src_position,
            self.transition_src_rotation,
            self.transition_dst_position,
            self.transition_dst_rotation,
            self.inertialize_blending_halflife,
            dt
        )

        self.simulation_positions_update(
            self.simulation_position,
            self.simulation_velocity,
            self.simulation_acceleration,
            self.desired_velocity,
            self.simulation_velocity_halflife,
            dt
        )

        self.simulation_rotations_update(
            self.simulation_rotation,
            self.simulation_angular_velocity,
            self.desired_rotation,
            self.simulation_rotation_halflife,
            dt
        )

        self.global_bone_positions, self.global_bone_rotations = forward_kinematics_full(
            self.bone_positions,
            self.bone_rotations,
            self.db.bone_parents
        )

        self._update_character()
        # self._update_skeletons()
        self._update_simulation_trajectory_objects()
        
        self.camera.set_target(self.global_bone_positions[0])

    def render(self):
        view_mat = self.renderer.camera.get_view_matrix()
        proj_mat = self.renderer.camera.get_projection_matrix()
        view_pos = self.renderer.camera.position

        for shader in self.programs.values():
            shader['view'].write(view_mat)
            shader['projection'].write(proj_mat)
            shader['view_pos'].write(view_pos)
        
        self.checkboard.render()

        if self.skeletons:
            for skeleton in self.skeletons:
                skeleton.render()
        
        if self.simulation_trajectory_objects:
            for obj in self.simulation_trajectory_objects:
                obj.render()

        if self.character is not None:
            self.character.render()
    
    def query_compute_trajectory_position_feature(self, query, offset, root_position, root_rotation, trajectory_positions):
        traj0 = quat.inv_mul_vec(root_rotation, trajectory_positions[1] - root_position)[[0, 2]]
        traj1 = quat.inv_mul_vec(root_rotation, trajectory_positions[2] - root_position)[[0, 2]]
        traj2 = quat.inv_mul_vec(root_rotation, trajectory_positions[3] - root_position)[[0, 2]]
        
        query[offset : offset + 6] = np.concatenate([traj0, traj1, traj2])
        

    def query_compute_trajectory_direction_feature(self, query, offset, root_rotation, trajectory_rotations):
        traj0 = quat.inv_mul_vec(root_rotation, quat.mul_vec(trajectory_rotations[1], np.array([0, 0, 1], dtype = np.float32)))[[0, 2]]
        traj1 = quat.inv_mul_vec(root_rotation, quat.mul_vec(trajectory_rotations[2], np.array([0, 0, 1], dtype = np.float32)))[[0, 2]]
        traj2 = quat.inv_mul_vec(root_rotation, quat.mul_vec(trajectory_rotations[3], np.array([0, 0, 1], dtype = np.float32)))[[0, 2]]

        query[offset : offset + 6] = np.concatenate([traj0, traj1, traj2])

    def desired_gait_update(self, desired_gait, desired_gait_velocity, dt, gait_change_halflife = 0.1):
        simple_spring_damper_exact(
            desired_gait,
            desired_gait_velocity,
            1.0 if self.renderer.controller.keys.get(glfw.KEY_SPACE) else 0.0,
            gait_change_halflife,
            dt
        )

    def desired_velocity_update(self, movement_input, simulation_rotation, fwrd_speed, side_speed, back_speed):
        global_stick_direction = movement_input
        
        local_stick_direction = quat.inv_mul_vec(simulation_rotation, global_stick_direction)

        if local_stick_direction[2] > 0:
            local_desired_velocity = np.array([side_speed, 0, fwrd_speed], dtype = np.float32) * local_stick_direction
        else:
            local_desired_velocity = np.array([side_speed, 0, back_speed], dtype = np.float32) * local_stick_direction
        
        return quat.mul_vec(simulation_rotation, local_desired_velocity)

    def desired_rotation_update(self, current_rotation, movement_input, desired_velocity):
        if length(movement_input) < 0.01: # Left Input이 없을 경우에는 Update 할 것이 없다.
            return current_rotation
        desired_direction = normalize(desired_velocity)
        return quat.from_angle_axis(np.arctan2(desired_direction[0], desired_direction[2]), np.array([0, 1, 0], dtype = np.float32))
    
    # 현재 Keyboard Input이 지속된다고 가정할 때 future trajectory, direction 계산
    # 모든 물리량은 Simulation Bone의 물리량

    def trajectory_desired_rotations_predict(self, desired_rotations, desired_velocities, desired_rotation, movement_input):
        desired_rotations[0, :] = desired_rotation
        for i in range(1, desired_rotations.shape[0]):
            desired_rotations[i, :] = self.desired_rotation_update(
                desired_rotations[i - 1, :],
                movement_input,
                desired_velocities[i]
            )
        
    def trajectory_rotations_predict(self, rotations, angular_velocities, rotation, angular_velocity, desired_rotations, halflife, dt):
        rotations[:] = rotation
        angular_velocities[:] = angular_velocity
        for i in range(1, rotations.shape[0]):
            self.simulation_rotations_update(
                rotations[i],
                angular_velocities[i],
                desired_rotations[i],
                halflife,
                i * dt
            )
    
    def trajectory_desired_velocities_predict(self, desired_velocities, trajectory_rotations, desired_velocity, movement_input, fwrd_speed, side_speed, back_speed, dt):
        desired_velocities[0] = desired_velocity
        
        for i in range(1, desired_velocities.shape[0]):
            # desired_velocities[i] = self.desired_velocity_update(
            #     movement_input,
            #     self.renderer.orbit_camera_update_azimuth(i * dt),
            #     trajectory_rotations[i],
            #     fwrd_speed,
            #     side_speed,
            #     back_speed
            # )

            desired_velocities[i] = self.desired_velocity_update(
                movement_input,
                trajectory_rotations[i],
                fwrd_speed,
                side_speed,
                back_speed
            )
    
    def trajectory_positions_predict(self, positions, velocities, accelerations, position, velocity, acceleration, desired_velocities, halflife, dt):
        positions[0] = position
        velocities[0] = velocity
        accelerations[0] = acceleration

        for i in range(1, positions.shape[0]):
            positions[i] = positions[i - 1]
            velocities[i] = velocities[i - 1]
            accelerations[i] = accelerations[i - 1]

            self.simulation_positions_update(
                positions[i],
                velocities[i],
                accelerations[i],
                desired_velocities[i],
                halflife,
                dt
            )

    # dt를 배치화해서 입력하면 배치 처리 가능
    def simulation_positions_update(self, position, velocity, acceleration, desired_velocity, halflife, dt):
        y = halflife_to_damping(halflife) / 2.0
        j0 = velocity - desired_velocity
        j1 = acceleration + j0 * y
        eydt = fast_negexp(y * dt)

        position[:] = eydt * ((-j1) / (y * y) + (-j0 - j1 * dt) / y) + (j1 / (y * y)) + j0 / y + desired_velocity * dt + position
        velocity[:] = eydt * (j0 + j1 * dt) + desired_velocity
        acceleration[:] = eydt * (acceleration - j1 * y * dt)
        
    def simulation_rotations_update(self, rotation, angular_velocity, desired_rotation, halflife, dt):
        simple_spring_damper_exact_quat(
            rotation,
            angular_velocity,
            desired_rotation,
            halflife,
            dt
        )

    def query_copy_denormalized_feature(self, query, db, feature, offset, size):
        query[offset : offset + size] = feature[offset : offset + size] * db.features_scale[offset : offset + size] + db.features_offset[offset : offset + size]
    
    def cleanup(self):
        for shader in self.programs.values():
            shader.release()

        self.checkboard.cleanup()

        if self.skeletons:
            for skeleton in self.skeletons:
                skeleton.cleanup()
        
        if self.character is not None:
            self.character.cleanup()

        if self.simulation_trajectory_objects:
            for obj in self.simulation_trajectory_objects:
                obj.cleanup()
   
    def _init_character(self, filename):
        character = load_character(filename)
        self.character = LBSObject(self.ctx, self.programs['lbs'],
                                   character['local_positions'], character['local_normals'], character['bone_weights'], character['bone_indices'], character['triangles'],
                                   color = self.character_color)
    
    def _init_skeletons(self):
        self.skeletons = []
        for i, parent in enumerate(self.db.bone_parents):
            if parent > 0: # NOT Simulation Bone & ROOT
                parent_position = glm.vec3(*self.global_bone_positions[parent])
                position = glm.vec3(*self.global_bone_positions[i])
                
                vertices, colors, normals, indices = create_cuboid(glm.length(position - parent_position), 0.08, 0.08, self.skeleton_color)
                skeleton = PhongObject(self.ctx, self.programs['phong'], vertices, colors, normals, indices)
                self.skeletons.append(skeleton)
    
    def _init_simulation_trajectory_objects(self):
        self.simulation_trajectory_objects = []
        for i in range(4):
            vertices, colors, normals, indices = create_sphere(radius = 0.08 + 0.002 * i, color = (0.7, 0.1 + i * 0.2, 0.1))
            self.simulation_trajectory_objects.append(PhongObject(self.ctx, self.programs['phong'], vertices, colors, normals, indices))

    def _update_character(self):
        bone_matrices = []

        for i, parent in enumerate(self.db.bone_parents):
            if parent != -1: # NOT Simulation Bone
                position = glm.vec3(*self.global_bone_positions[i])
                rotation = glm.quat(*self.global_bone_rotations[i])
                bone_matrix = glm.mat4(rotation)
                bone_matrix[3] = glm.vec4(position, 1.0)
                bone_matrices.append(bone_matrix)

        self.character.update(bone_matrices)

    def _update_skeletons(self):
        for i, parent in enumerate(self.db.bone_parents):
            if parent > 0: # NOT Simulation Bone & ROOT
                parent_position = glm.vec3(*self.global_bone_positions[parent])
                parent_rotation = glm.quat(*self.global_bone_rotations[parent])
                position = glm.vec3(*self.global_bone_positions[i])
                origin = (parent_position + position) / 2
                x_axis = glm.normalize(position - parent_position)
                z_axis = glm.normalize(glm.cross(x_axis, parent_rotation * glm.vec3(0, 1.0, 0)))
                y_axis = glm.normalize(glm.cross(z_axis, x_axis))
                model = glm.mat4(glm.vec4(x_axis, 0), glm.vec4(y_axis, 0), glm.vec4(z_axis, 0), glm.vec4(origin, 1.0))
                
                self.skeletons[i - 2].update(model) # ignore Simulation Bone & ROOT
    
    def _update_simulation_trajectory_objects(self):
        for i, obj in enumerate(self.simulation_trajectory_objects):
            model = glm.mat4(1.0)
            pos = glm.vec3(self.trajectory_positions[i])
            model[3] = glm.vec4(pos, 1.0)
            obj.update(model)

    def _read_glsl_file(self, shader_path):
        with open(shader_path, 'r', encoding = 'utf-8') as shader_file:
            shader_code = shader_file.read()
            return shader_code

    def _init_shaders(self):
        self.programs = {
            'checkerboard': self.ctx.program(
                self._read_glsl_file('./shader/checkerboard_vert_shader.glsl'),
                self._read_glsl_file('./shader/checkerboard_frag_shader.glsl')
            ),
            'phong': self.ctx.program(
                self._read_glsl_file('./shader/phong_vert_shader.glsl'),
                self._read_glsl_file('./shader/phong_frag_shader.glsl')
            ),
            'lbs': self.ctx.program(
                self._read_glsl_file('./shader/lbs_vert_shader.glsl'),
                self._read_glsl_file('./shader/lbs_frag_shader.glsl')
            )
        }

        # # 광원 설정
        # for shader in self.programs.values():
        #     shader['light_pos'].write(glm.vec3(0.0, 10.0, 0.0))
        #     shader['light_color'].write(glm.vec3(1.0, 1.0, 1.0))

        # # 재질 속성 변경
        # self.programs['phong']['ambient_strength'].value = 0.1
        # self.programs['phong']['specular_strength'].value = 0.3
        # self.programs['phong']['shininess'].value = 16.0
    
    