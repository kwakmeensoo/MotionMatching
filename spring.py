from common import *
import numpy as np
import quat

def damper_exact(x, g, halflife, dt, eps=1e-5):
    """
    x: array[..., 3]
    g: array[..., 3]
    """
    return lerp(x, g, 1.0 - fast_negexp((LN2 * dt) / (halflife + eps)))

def damper_exact_quat(x, g, halflife, dt, eps=1e-5):
    """
    x: array[..., 4]
    g: array[..., 4]
    """
    return quat.slerp_shortest_approx(x, g, 1.0 - fast_negexp((LN2 * dt) / (halflife + eps)))

def damper_adjustment_exact(g, halflife, dt, eps=1e-5):
    """g: array[..., 3]"""
    return g * (1.0 - fast_negexp((LN2 * dt) / (halflife + eps)))

def damper_adjustment_exact_quat(g, halflife, dt, eps=1e-5):
    """g: array[..., 4]"""
    return quat.slerp_shortest_approx(quat.eye(g.shape), g, 1.0 - fast_negexp((LN2 * dt) / (halflife + eps)))

def halflife_to_damping(halflife, eps=1e-5):
    return (4.0 * LN2) / (halflife + eps)

def damping_to_halflife(damping, eps=1e-5):
    return (4.0 * LN2) / (damping + eps)

def frequency_to_stiffness(frequency):
    return (2.0 * PI * frequency) ** 2

def stiffness_to_frequency(stiffness):
    return np.sqrt(stiffness) / (2.0 * PI)

def simple_spring_damper_exact(x, v, x_goal, halflife, dt):
    """
    In-place update of spring-damper system for vectors.
    
    Parameters:
        x: array[..., 3] - current position (modified in-place)
        v: array[..., 3] - current velocity (modified in-place)
        x_goal: array[..., 3] - goal position
        halflife: float or array[...] - halflife constant
        dt: float or array[...] - timestep
    """
    y = halflife_to_damping(halflife) / 2.0
    
    # Compute intermediates using temporary arrays
    j0 = x - x_goal
    j1 = v + j0 * y
    eydt = fast_negexp(y * dt)
    
    # In-place updates
    x[...] = eydt * (j0 + j1 * dt) + x_goal
    v[...] = eydt * (v - j1 * y * dt)

def simple_spring_damper_exact_quat(x, v, x_goal, halflife, dt):
    """
    In-place update of spring-damper system for quaternions.
    
    Parameters:
        x: array[..., 4] - current rotation (modified in-place)
        v: array[..., 3] - current angular velocity (modified in-place)
        x_goal: array[..., 4] - goal rotation
        halflife: float or array[...] - halflife constant
        dt: float or array[...] - timestep
    """
    y = halflife_to_damping(halflife) / 2.0
    
    j0 = quat.to_scaled_angle_axis(quat.abs(quat.mul(x, quat.inv(x_goal))))
    j1 = v + j0 * y
    eydt = fast_negexp(y * dt)
    
    x[...] = quat.mul(quat.from_scaled_angle_axis(eydt * (j0 + j1 * dt)), x_goal)
    v[...] = eydt * (v - j1 * y * dt)

def decay_spring_damper_exact(x, v, halflife, dt):
    """
    In-place decay spring damper update for vectors.
    
    Parameters:
        x: array[..., 3] - position (modified in-place)
        v: array[..., 3] - velocity (modified in-place)
        halflife: float or array[...] - halflife constant
        dt: float or array[...] - timestep
    """
    y = halflife_to_damping(halflife) / 2.0

    j1 = v + x * y
    eydt = fast_negexp(y * dt)
    
    x[...] = eydt * (x + j1 * dt)
    v[...] = eydt * (v - j1 * y * dt)

def decay_spring_damper_exact_quat(x, v, halflife, dt):
    """
    In-place decay spring damper update for quaternions.
    
    Parameters:
        x: array[..., 4] - rotation (modified in-place)
        v: array[..., 3] - angular velocity (modified in-place)
        halflife: float or array[...] - halflife constant
        dt: float or array[...] - timestep
    """
    y = halflife_to_damping(halflife) / 2.0
    
    j0 = quat.to_scaled_angle_axis(x)
    j1 = v + j0 * y
    eydt = fast_negexp(y * dt)
    
    x[...] = quat.from_scaled_angle_axis(eydt * (j0 + j1 * dt))
    v[...] = eydt * (v - j1 * y * dt)

def inertialize_transition(off_x, off_v, src_x, src_v, dst_x, dst_v):
    """
    In-place inertialize transition for vectors.
    
    Parameters:
        off_x: array[..., 3] - position offset (modified in-place)
        off_v: array[..., 3] - velocity offset (modified in-place)
        src_x, src_v, dst_x, dst_v: array[..., 3]
    """
    off_x[...] = (src_x + off_x) - dst_x
    off_v[...] = (src_v + off_v) - dst_v

def inertialize_transition_quat(off_x, off_v, src_x, src_v, dst_x, dst_v):
    """
    In-place inertialize transition for quaternions.
    
    Parameters:
        off_x: array[..., 4] - rotation offset (modified in-place)
        off_v: array[..., 3] - angular velocity offset (modified in-place)
        src_x, dst_x: array[..., 4]
        src_v, dst_v: array[..., 3]
    """
    off_x[...] = quat.abs(quat.mul(quat.mul(off_x, src_x), quat.inv(dst_x)))
    off_v[...] = (src_v + off_v) - dst_v

def inertialize_update(out_x, out_v, off_x, off_v, in_x, in_v, halflife, dt):
    """
    In-place inertialize update for vectors.
    
    Parameters:
        out_x: array[..., 3] - output position (modified in-place)
        out_v: array[..., 3] - output velocity (modified in-place)
        off_x: array[..., 3] - position offset (modified in-place)
        off_v: array[..., 3] - velocity offset (modified in-place)
        in_x: array[..., 3] - input position
        in_v: array[..., 3] - input velocity
    """
    decay_spring_damper_exact(off_x, off_v, halflife, dt)
    out_x[...] = in_x + off_x
    out_v[...] = in_v + off_v

def inertialize_update_quat(out_x, out_v, off_x, off_v, in_x, in_v, halflife, dt):
    """
    In-place inertialize update for quaternions.
    
    Parameters:
        out_x: array[..., 4] - output rotation (modified in-place)
        out_v: array[..., 3] - output angular velocity (modified in-place)
        off_x: array[..., 4] - rotation offset (modified in-place)
        off_v: array[..., 3] - angular velocity offset (modified in-place)
        in_x: array[..., 4] - input rotation
        in_v: array[..., 3] - input angular velocity
    """
    decay_spring_damper_exact_quat(off_x, off_v, halflife, dt)
    out_x[...] = quat.mul(off_x, in_x)
    out_v[...] = off_v + quat.mul_vec(off_x, in_v)