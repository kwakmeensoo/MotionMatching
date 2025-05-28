from common import *
import numpy as np

def load_character(filename):
    with open(filename, 'rb') as f:
        positions = read_array_1d_vec3(f)
        normals = read_array_1d_vec3(f)
        texcoords = read_array_1d_vec2(f)
        triangles = read_array_1d(f, dtype = np.uint16)

        bone_weights = read_array_2d(f)
        bone_indices = read_array_2d(f, dtype = np.uint16) # important: saved in one-base.

        bone_rest_positions = read_array_1d_vec3(f)
        bone_rest_rotations = read_array_1d_quat(f)

        local_positions = np.zeros((*bone_indices.shape, 3), dtype = np.float32)
        local_normals = np.zeros((*bone_indices.shape, 3), dtype = np.float32)

        for i, indices in enumerate(bone_indices):
            for j, index in enumerate(indices):
                vertex_rest_position = positions[i]
                vertex_rest_normal = normals[i]
                bone_rest_position = bone_rest_positions[index, :]
                bone_rest_rotation = bone_rest_rotations[index, :]

                local_positions[i, j] = quat_inv_mul_vec(bone_rest_rotation, vertex_rest_position - bone_rest_position)
                local_normals[i, j] = quat_inv_mul_vec(bone_rest_rotation, vertex_rest_normal)
        
        # one-base -> zero-base
        bone_indices -= 1
        
        return {
            'local_positions': local_positions,
            'local_normals': local_normals,
            'bone_weights' : bone_weights,
            'bone_indices' : bone_indices,
            'triangles' : triangles,
            'texcoords' : texcoords
        }
        
def _fast_cross(x, y):
    return np.concatenate([
        x[..., 1:2] * y[..., 2:3] - x[..., 2:3] * y[..., 1:2],
        x[..., 2:3] * y[..., 0:1] - x[..., 0:1] * y[..., 2:3],
        x[..., 0:1] * y[..., 1:2] - x[..., 1:2] * y[..., 0:1]
    ], axis = -1)

def quat_inv_mul_vec(q, x):
   q = np.asarray([1, -1, -1, -1], dtype = np.float32) * q
   t = 2.0 * _fast_cross(q[..., 1:], x)
   return x + q[..., 0][..., np.newaxis] * t + _fast_cross(q[..., 1:], t)
