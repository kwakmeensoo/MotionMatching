import numpy as np

PI = 3.14159265358979323846
LN2 = 0.69314718056

# Utility Functions

def length(x):
    return np.linalg.norm(x)

def cross(a, b):
    return np.concatenate([
        a[...,1:2]*b[...,2:3] - a[...,2:3]*b[...,1:2],
        a[...,2:3]*b[...,0:1] - a[...,0:1]*b[...,2:3],
        a[...,0:1]*b[...,1:2] - a[...,1:2]*b[...,0:1]], axis=-1)

def clamp(x, min_val, max_val):
    return np.clip(x, min_val, max_val)

def fast_negexp(x):
    return 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)

def fast_atan(x):
    z = np.abs(x)
    w = np.where(z > 1.0, 1.0 / z, z)
    y = (PI / 4.0) * w - w * (w - 1.0) * (0.2447 + 0.0663 * w)
    return np.copysign(np.where(z > 1.0, PI / 2.0 - y, y), x)

def lerp(x, y, a):
    return (1.0 - a) * x + a * y

def normalize(v, eps = 1e-8):
    return v / (np.sqrt(np.sum(v * v, axis = -1))[..., np.newaxis] + eps)

# Binary File Read

def read_array_1d_vec2(f, dtype = np.float32):
    size = np.fromfile(f, np.int32, 1)[0]
    return np.fromfile(f, dtype, size * 2).reshape(-1, 2)

def read_array_1d_vec3(f, dtype=np.float32):
    size = np.fromfile(f, np.int32, 1)[0]
    return np.fromfile(f, dtype, size * 3).reshape(-1, 3)

def read_array_1d_quat(f, dtype = np.float32):
    size = np.fromfile(f, np.int32, 1)[0]
    return np.fromfile(f, dtype, size * 4).reshape(-1, 4)

def read_array_1d(f, dtype = np.float32):
    size = np.fromfile(f, np.int32, 1)[0]
    return np.fromfile(f, dtype, size)

def read_array_2d_vec2(f, dtype = np.float32):
    shape = np.fromfile(f, np.int32, 2)
    return np.fromfile(f, dtype, shape[0] * shape[1] * 2).reshape(shape[0], shape[1], 2)

def read_array_2d_vec3(f, dtype = np.float32):
    shape = np.fromfile(f, np.int32, 2)
    return np.fromfile(f, dtype, shape[0] * shape[1] * 3).reshape(shape[0], shape[1], 3)

def read_array_2d_quat(f, dtype = np.float32):
    shape = np.fromfile(f, np.int32, 2)
    return np.fromfile(f, dtype, shape[0] * shape[1] * 4).reshape(shape[0], shape[1], 4)

def read_array_2d(f, dtype = np.float32):
    shape = np.fromfile(f, np.int32, 2)
    return np.fromfile(f, dtype, shape[0] * shape[1]).reshape(shape)
