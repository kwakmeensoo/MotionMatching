import numpy as np

# quat: ndarray[..., 4]
# vec : ndarray[..., 3]

def _fast_cross(a, b):
    return np.concatenate([
        a[...,1:2]*b[...,2:3] - a[...,2:3]*b[...,1:2],
        a[...,2:3]*b[...,0:1] - a[...,0:1]*b[...,2:3],
        a[...,0:1]*b[...,1:2] - a[...,1:2]*b[...,0:1]], axis=-1)

def eye(shape, dtype=np.float32):
    return np.ones(list(shape) + [4], dtype=dtype) * np.asarray([1, 0, 0, 0], dtype=dtype)

def dot(x, y):
    return np.sum(x * y, axis = -1)

def length(x):
    return np.sqrt(np.sum(x * x, axis=-1))

def normalize(x, eps=1e-8):
    return x / (length(x)[...,np.newaxis] + eps)

def abs(x):
    return np.where(x[...,0:1] > 0.0, x, -x)

def from_angle_axis(angle, axis):
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q

def to_xform(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)[...,np.newaxis,:],
        np.concatenate([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)[...,np.newaxis,:],
        np.concatenate([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)[...,np.newaxis,:],
    ], axis=-2)

def to_xform_xy(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz], axis=-1)[...,np.newaxis,:],
        np.concatenate([xy + wz, 1.0 - (xx + zz)], axis=-1)[...,np.newaxis,:],
        np.concatenate([xz - wy, yz + wx], axis=-1)[...,np.newaxis,:],
    ], axis=-2)

def from_euler(e, order='zyx'):
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = from_angle_axis(e[..., 0], axis[order[0]])
    q1 = from_angle_axis(e[..., 1], axis[order[1]])
    q2 = from_angle_axis(e[..., 2], axis[order[2]])

    return mul(q0, mul(q1, q2))

def from_xform(ts):

    return normalize(
        np.where((ts[...,2,2] < 0.0)[...,np.newaxis],
            np.where((ts[...,0,0] >  ts[...,1,1])[...,np.newaxis],
                np.concatenate([
                    (ts[...,2,1]-ts[...,1,2])[...,np.newaxis],
                    (1.0 + ts[...,0,0] - ts[...,1,1] - ts[...,2,2])[...,np.newaxis],
                    (ts[...,1,0]+ts[...,0,1])[...,np.newaxis],
                    (ts[...,0,2]+ts[...,2,0])[...,np.newaxis]], axis=-1),
                np.concatenate([
                    (ts[...,0,2]-ts[...,2,0])[...,np.newaxis],
                    (ts[...,1,0]+ts[...,0,1])[...,np.newaxis],
                    (1.0 - ts[...,0,0] + ts[...,1,1] - ts[...,2,2])[...,np.newaxis],
                    (ts[...,2,1]+ts[...,1,2])[...,np.newaxis]], axis=-1)),
            np.where((ts[...,0,0] < -ts[...,1,1])[...,np.newaxis],
                np.concatenate([
                    (ts[...,1,0]-ts[...,0,1])[...,np.newaxis],
                    (ts[...,0,2]+ts[...,2,0])[...,np.newaxis],
                    (ts[...,2,1]+ts[...,1,2])[...,np.newaxis],
                    (1.0 - ts[...,0,0] - ts[...,1,1] + ts[...,2,2])[...,np.newaxis]], axis=-1),
                np.concatenate([
                    (1.0 + ts[...,0,0] + ts[...,1,1] + ts[...,2,2])[...,np.newaxis],
                    (ts[...,2,1]-ts[...,1,2])[...,np.newaxis],
                    (ts[...,0,2]-ts[...,2,0])[...,np.newaxis],
                    (ts[...,1,0]-ts[...,0,1])[...,np.newaxis]], axis=-1))))

def from_xform_xy(x):

    c2 = _fast_cross(x[...,0], x[...,1])
    c2 = c2 / np.sqrt(np.sum(np.square(c2), axis=-1))[...,np.newaxis]
    c1 = _fast_cross(c2, x[...,0])
    c1 = c1 / np.sqrt(np.sum(np.square(c1), axis=-1))[...,np.newaxis]
    c0 = x[...,0]

    return from_xform(np.concatenate([
        c0[...,np.newaxis],
        c1[...,np.newaxis],
        c2[...,np.newaxis]], axis=-1))

def inv(q):
    return np.asarray([1, -1, -1, -1], dtype=np.float32) * q

def mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

def inv_mul(x, y):
    return mul(inv(x), y)

def mul_inv(x, y):
    return mul(x, inv(y))

def mul_vec(q, x):
    t = 2.0 * _fast_cross(q[..., 1:], x)
    return x + q[..., 0][..., np.newaxis] * t + _fast_cross(q[..., 1:], t)

def inv_mul_vec(q, x):
    return mul_vec(inv(q), x)

def unroll(x):
    y = x.copy()
    for i in range(1, len(x)):
        d0 = np.sum( y[i] * y[i-1], axis=-1)
        d1 = np.sum(-y[i] * y[i-1], axis=-1)
        y[i][d0 < d1] = -y[i][d0 < d1]
    return y

def between(x, y):
    return np.concatenate([
        np.sqrt(np.sum(x*x, axis=-1) * np.sum(y*y, axis=-1))[...,np.newaxis] +
        np.sum(x * y, axis=-1)[...,np.newaxis],
        _fast_cross(x, y)], axis=-1)

def log(x, eps=1e-5):
    length = np.sqrt(np.sum(np.square(x[...,1:]), axis=-1))[...,np.newaxis] + 1e-8
    halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, x[...,0:1]) / length)
    return halfangle * x[...,1:]

def exp(x, eps=1e-5):
    halfangle = np.sqrt(np.sum(np.square(x), axis=-1))[...,np.newaxis]
    c = np.where(halfangle < eps, np.ones_like(halfangle), np.cos(halfangle))
    s = np.where(halfangle < eps, np.ones_like(halfangle), np.sinc(halfangle / np.pi))
    return np.concatenate([c, s * x], axis=-1)

def to_scaled_angle_axis(x, eps=1e-5):
    return 2.0 * log(x, eps)

def from_scaled_angle_axis(x, eps=1e-5):
    return exp(x / 2.0, eps)

def angle_between(x, y):
    diff = abs(mul_inv(x, y))
    return 2.0 * np.arccos(np.clip(diff[..., 0], -1.0, 1.0))

def between(x, y):
    c = _fast_cross(x, y)
    w = np.sqrt(np.sum(x * x, axis = -1) * np.sum(y * y, axis = -1)) + np.sum(x * y, axis = -1)
    return normalize(np.concatenate([w[..., np.newaxis], c], axis = -1))

def differentiate_angular_velocity(next_quat, curr_quat, dt, eps = 1e-8):
    return to_scaled_angle_axis(abs(mul(next_quat, inv(curr_quat))), eps) / dt

def fk(lrot, lpos, parents):

    gp, gr = [lpos[...,:1,:]], [lrot[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(mul    (gr[parents[i]], lrot[...,i:i+1,:]))

    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)

def ik(grot, gpos, parents):

    return (
        np.concatenate([
            grot[...,:1,:],
            mul(inv(grot[...,parents[1:],:]), grot[...,1:,:]),
        ], axis=-2),
        np.concatenate([
            gpos[...,:1,:],
            mul_vec(
                inv(grot[...,parents[1:],:]),
                gpos[...,1:,:] - gpos[...,parents[1:],:]),
        ], axis=-2))

def fk_vel(lrot, lpos, lvel, lang, parents):

    gp, gr, gv, ga = [lpos[...,:1,:]], [lrot[...,:1,:]], [lvel[...,:1,:]], [lang[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(mul    (gr[parents[i]], lrot[...,i:i+1,:]))
        gv.append(mul_vec(gr[parents[i]], lvel[...,i:i+1,:]) +
            _fast_cross(ga[parents[i]], mul_vec(gr[parents[i]], lpos[...,i:i+1,:])) +
            gv[parents[i]])
        ga.append(mul_vec(gr[parents[i]], lang[...,i:i+1,:]) + ga[parents[i]])

    return (
        np.concatenate(gr, axis=-2),
        np.concatenate(gp, axis=-2),
        np.concatenate(gv, axis=-2),
        np.concatenate(ga, axis=-2))

# quat to euler, in radian
def to_euler(q, order='xyz'):
    assert q.shape[-1] == 4
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    epsilon=0
    if order == 'xyz':
        x = np.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = np.asin(np.clip(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = np.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = np.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = np.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = np.asin(np.clip(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = np.asin(np.clip(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = np.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = np.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = np.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = np.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = np.asin(np.clip(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = np.asin(np.clip(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = np.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = np.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = np.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = np.asin(np.clip(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = np.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise
    resdict = {"x":x, "y":y, "z":z}
    reslist = [resdict[order[i]] for i in range(len(order))]
    return np.stack(reslist, axis=-1)

def nlerp(x, y, a):
    return normalize(
        x * (1.0 - a)[..., np.newaxis] +
        y * a[..., np.newaxis]
    )

def nlerp_shortest(x, y, a):
    d = dot(x, y)
    y = np.where(d[..., np.newaxis] < 0.0, -y, y)
    return nlerp(x, y, a)

def slerp_shortest(x, y, a, eps = 1e-5):
    d = np.clip(dot(x, y), -1.0, 1.0)
    y = np.where(d[..., np.newaxis] < 0.0, -y, y)

    angle = np.arccos(np.abs(d))

    mask = angle < eps
    if np.any(mask):
        return np.where(
            mask[..., np.newaxis],
            nlerp(x, y, a),
            x * (np.sin((1.0 - a) * angle) / np.sin(angle))[..., np.newaxis] +
            y * (np.sin(a * angle) / np.sin(angle))[..., np.newaxis]
        )
    return x * (np.sin((1.0 - a) * angle) / np.sin(angle))[..., np.newaxis] + y * (np.sin(a * angle) / np.sin(angle))[..., np.newaxis]

def slerp_shortest_approx(x, y, a):
    ca = dot(x, y)

    y = np.where(ca[..., np.newaxis] < 0.0, -y, y)
    ca = np.abs(ca)

    a1 = 1.0904 + ca * (-3.2452 + ca * (3.55645 - ca * 1.43519))
    a2 = 0.848013 + ca * (-1.06021 + ca * 0.215638)

    k = a1 * (a - 0.5) * (a - 0.5) + a2
    adjusted_a = a + a * (a - 0.5) * (a - 1.0) * k

    return nlerp(x, y, adjusted_a)