import numpy as np
import moderngl as mgl
import glm
import bvh_utils
from camera import Camera, CameraMode
from geometry import *
from objects import *
from lbs import load_character

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

        self.is_playing = False
        self.speed = 1.0

        self.frame_time = None
        self.curr_frame = 0
        self.num_frames = 1
        self.skeletons = None
        self.character = None

        self.bvh_parents = None
        self.bvh_positions = None
        self.bvh_rotations = None
        self.bvh_order = None
        self.global_positions: list[glm.vec3] = []
        self.global_rotations: list[glm.quat] = []

    def update(self):
        if self.is_playing:
            self.curr_frame = min(self.curr_frame + 1, self.num_frames - 1)
        
        if self.bvh_parents is not None:
            self._forward_kinematics(self.bvh_parents, self.bvh_positions[self.curr_frame], self.bvh_rotations[self.curr_frame], self.bvh_order)

        if self.skeletons:
            self._update_skeletons()
                
            if self.camera.mode == CameraMode.ORBIT:
                self.camera.set_target(self.bvh_positions[self.curr_frame][0])
        
        if self.character is not None:
            self._update_character()

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

        if self.character is not None:
            self.character.render()
    
    def cleanup(self):
        for shader in self.programs.values():
            shader.release()

        self.checkboard.cleanup()

        if self.skeletons:
            for skeleton in self.skeletons:
                skeleton.cleanup()
        
        if self.character is not None:
            self.character.cleanup()
    
    def update_speed(self):
        self.renderer.set_frame_time(self.frame_time / self.speed)
        
    def init_bvh(self, filename):
        if self.skeletons:
            for skeleton in self.skeletons:
                skeleton.cleanup()
        
        if self.character is not None:
            self.character.cleanup()
        
        self.skeletons = []
        self.character = None
        
        bvh_data = bvh_utils.load(filename)
        # cm -> m
        self.bvh_parents = bvh_data['parents']
        self.bvh_positions = bvh_data['positions'] / 100.0
        self.bvh_rotations = bvh_data['rotations']
        self.bvh_order = bvh_data['order']

        self.frame_time = bvh_data['frame_time']
        self.renderer.set_frame_time(self.frame_time)
        self.num_frames = bvh_data['positions'].shape[0]
        self.curr_frame = 0

        self.is_playing = False
        self.speed = 1.0
        self.update_speed()
        
        self._forward_kinematics(self.bvh_parents, self.bvh_positions[0], self.bvh_rotations[0], self.bvh_order)

        self._init_skeletons()
        self._update_skeletons()

        self._init_character('character.bin')
        self._update_character()

        translation_mat = glm.mat4(1.0)
        translation_mat[3] = glm.vec4(2.5, 0.0, 0.0, 1.0)
        self.character.update(model = translation_mat)

        if self.camera.mode == CameraMode.ORBIT:
            self.camera.set_target(self.bvh_positions[0][0])
    
    def _forward_kinematics(self, parents, positions, rotations, order):
        self.global_positions = []
        self.global_rotations = []
        for i, parent in enumerate(parents):
            if parent == -1:
                self.global_positions.append(glm.vec3(positions[i]))
                self.global_rotations.append(self._quat_from_euler(rotations[i], order))
            else:
                global_position = self.global_positions[parent] + self.global_rotations[parent] * glm.vec3(positions[i])
                global_rotation = self.global_rotations[parent] * self._quat_from_euler(rotations[i], order)
                self.global_positions.append(global_position)
                self.global_rotations.append(global_rotation)

    def _init_character(self, filename):
        character = load_character(filename)
        self.character = LBSObject(self.ctx, self.programs['lbs'],
                                   character['local_positions'], character['local_normals'], character['bone_weights'], character['bone_indices'], character['triangles'],
                                   color = (0.9, 0.2, 0.2))
    
    def _init_skeletons(self):
        for i, parent in enumerate(self.bvh_parents):
            if parent != -1:
                parent_position = self.global_positions[parent]
                position = self.global_positions[i]
                
                vertices, colors, normals, indices = create_cuboid(glm.length(position - parent_position), 0.08, 0.08, self.skeleton_color)
                skeleton = PhongObject(self.ctx, self.programs['phong'], vertices, colors, normals, indices)
                self.skeletons.append(skeleton)

    def _update_character(self):
        bone_matrices = []

        for i, parent in enumerate(self.bvh_parents):
            position = self.global_positions[i]
            rotation = self.global_rotations[i]
            bone_matrix = glm.mat4(rotation)
            bone_matrix[3] = glm.vec4(position, 1.0)
            bone_matrices.append(bone_matrix)

        self.character.update(bone_matrices)

    def _update_skeletons(self):
        for i, parent in enumerate(self.bvh_parents):
            if parent != -1:
                parent_position = self.global_positions[parent]
                parent_rotation = self.global_rotations[parent]
                position = self.global_positions[i]
                origin = (parent_position + position) / 2
                x_axis = glm.normalize(position - parent_position)
                z_axis = glm.normalize(glm.cross(x_axis, parent_rotation * glm.vec3(0, 1.0, 0)))
                y_axis = glm.normalize(glm.cross(z_axis, x_axis))
                model = glm.mat4(glm.vec4(x_axis, 0), glm.vec4(y_axis, 0), glm.vec4(z_axis, 0), glm.vec4(origin, 1.0))
                
                self.skeletons[i - 1].update(model)

    def _quat_from_euler(self, angles, order):
        rad_angles = [glm.radians(angle) for angle in angles]
        
        axis_map = {
            'x': glm.vec3(1.0, 0.0, 0.0),
            'y': glm.vec3(0.0, 1.0, 0.0),
            'z': glm.vec3(0.0, 0.0, 1.0)
        }

        q = glm.quat()

        for i, axis in enumerate(order):
            axis_quat = glm.angleAxis(rad_angles[i], axis_map[axis])
            q = q * axis_quat

        return q
            

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
    
    