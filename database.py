from common import *
import numpy as np
import quat

BOUND_SM_SIZE = 16
BOUND_LR_SIZE = 64

class Database:
    def __init__(self, filename):
        # N : # of frames
        # J : # of bones
        # R : # of ranges
        # F : # of features
        with open(filename, 'rb') as f:
            self.bone_positions  = read_array_2d_vec3(f)
            self.bone_velocities = read_array_2d_vec3(f)
            self.bone_rotations  = read_array_2d_quat(f)
            self.bone_angular_velocities = read_array_2d_vec3(f)
            self.bone_parents    = read_array_1d(f, dtype = np.int32)

            self.range_starts = read_array_1d(f, dtype = np.int32)
            self.range_stops  = read_array_1d(f, dtype = np.int32)

        self.features : np.ndarray = None # (N, F)
        self.features_offset : np.ndarray = None # (F)
        self.features_scale  : np.ndarray = None # (F)

        self.bound_sm_min : np.ndarray = None
        self.bound_sm_max : np.ndarray = None
        self.bound_lr_min : np.ndarray = None
        self.bound_lr_max : np.ndarray = None

    def nframes(self):
        return self.bone_positions.shape[0]
    
    def nbones(self):
        return self.bone_positions.shape[1]
    
    def nranges(self):
        return self.range_starts.shape[0]
    
    def nfeatures(self):
        return self.features.shape[1]
    
    def clamp_trajectory_index(self, frame, offset):
        for (start, stop) in zip(self.range_starts, self.range_stops):
            if start <= frame and frame < stop:
                return clamp(frame + offset, start, stop - 1)
    
def normalize_feature(features, features_offset, features_scale, offset, size, weight = 1.0):
    # features를 직접 변경
    features_offset[offset : offset + size] = np.mean(features[:, offset : offset + size], axis = 0)
    vars = np.var(features[:, offset : offset + size], axis = 0)
    std = np.mean(np.sqrt(vars))

    features_scale[offset : offset + size] = std / weight

    features[:, offset : offset + size] = (features[:, offset : offset + size] - features_offset[offset : offset + size]) / features_scale[offset : offset + size]

def denormalize_feature(features, features_offset, features_scale):
    # features를 직접 변경
    features[...] = features * features_scale + features_offset

# --- kinematics ---
# 배치 연산을 지원하는 forward kinematics

def forward_kinematics(bone_positions, bone_rotations, bone_parents, bone):
    if bone_parents[bone] != -1:
        parent_position, parent_rotation = forward_kinematics(bone_positions, bone_rotations, bone_parents, bone_parents[bone])
        return (quat.mul_vec(parent_rotation, bone_positions[..., bone, :]) + parent_position,
                quat.mul(parent_rotation, bone_rotations[..., bone, :]))
    else:
        return bone_positions[..., bone, :], bone_rotations[..., bone, :]

def forward_kinematics_velocity(bone_positions, bone_velocities, bone_rotations, bone_angular_velocities, bone_parents, bone):
    if bone_parents[bone] != -1:
        parent_position, parent_velocity, parent_rotation, parent_angular_velocity = \
            forward_kinematics_velocity(bone_positions, bone_velocities, bone_rotations, bone_angular_velocities, bone_parents, bone_parents[bone])

        return (
            quat.mul_vec(parent_rotation, bone_positions[..., bone, :]) + parent_position,
            parent_velocity + quat.mul_vec(parent_rotation, bone_velocities[..., bone, :]) + cross(parent_angular_velocity, quat.mul_vec(parent_rotation, bone_positions[..., bone, :])),
            quat.mul(parent_rotation, bone_rotations[..., bone, :]),
            quat.mul_vec(parent_rotation, bone_angular_velocities[..., bone, :]) + parent_angular_velocity
        )
    else:
        return (
            bone_positions[..., bone, :],
            bone_velocities[..., bone, :],
            bone_rotations[..., bone, :],
            bone_angular_velocities[..., bone, :]
        )

def forward_kinematics_full(bone_positions, bone_rotations, bone_parents):
    global_bone_positions = np.zeros_like(bone_positions)
    global_bone_rotations = np.zeros_like(bone_rotations)
    
    for i, par in enumerate(bone_parents):
        if par == -1:
            global_bone_positions[..., i, :] = bone_positions[..., i, :]
            global_bone_rotations[..., i, :] = bone_rotations[..., i, :]
        else:
            parent_position = global_bone_positions[..., par, :]
            parent_rotation = global_bone_rotations[..., par, :]
            global_bone_positions[..., i, :] = quat.mul_vec(parent_rotation, bone_positions[..., i, :]) + parent_position
            global_bone_rotations[..., i, :] = quat.mul(parent_rotation, bone_rotations[..., i, :])
    
    return global_bone_positions, global_bone_rotations

# --- feature calculation ---
# 배치 연산을 통해 각 프레임에서 지정된 bone의 position, velocity 등의 feature를 추출한다.
# 이후, normalize
def compute_bone_position_feature(db, offset, bone, weight = 1.0):
    # (N, 3)
    bone_position, bone_rotation = forward_kinematics(db.bone_positions, db.bone_rotations, db.bone_parents, bone)
    bone_position = quat.mul_vec(quat.inv(db.bone_rotations[..., 0, :]), bone_position - db.bone_positions[..., 0, :])
    
    db.features[..., offset : offset + 3] = bone_position[...]
    normalize_feature(db.features, db.features_offset, db.features_scale, offset, 3, weight)

def compute_bone_velocity_feature(db, offset, bone, weight = 1.0):
    # (N, 3)
    bone_position, bone_velocity, bone_rotation, bone_angular_velocity = \
        forward_kinematics_velocity(db.bone_positions, db.bone_velocities, db.bone_rotations, db.bone_angular_velocities, db.bone_parents, bone)
    bone_velocity = quat.mul_vec(quat.inv(db.bone_rotations[..., 0, :]), bone_velocity)
    
    db.features[..., offset : offset + 3] = bone_velocity[...]
    normalize_feature(db.features, db.features_offset, db.features_scale, offset, 3, weight)

def compute_trajectory_position_feature(db: Database, offset, weight = 1.0):
    for i, feature in enumerate(db.features):
        t0 = db.clamp_trajectory_index(i, 20)
        t1 = db.clamp_trajectory_index(i, 40)
        t2 = db.clamp_trajectory_index(i, 60)

        trajectory_pos0 = quat.mul_vec(quat.inv(db.bone_rotations[i, 0, :]), db.bone_positions[t0, 0, :] - db.bone_positions[i, 0, :])[[0, 2]]
        trajectory_pos1 = quat.mul_vec(quat.inv(db.bone_rotations[i, 0, :]), db.bone_positions[t1, 0, :] - db.bone_positions[i, 0, :])[[0, 2]]
        trajectory_pos2 = quat.mul_vec(quat.inv(db.bone_rotations[i, 0, :]), db.bone_positions[t2, 0, :] - db.bone_positions[i, 0, :])[[0, 2]]

        feature[offset : offset + 6] = np.concatenate([trajectory_pos0, trajectory_pos1, trajectory_pos2])

    normalize_feature(db.features, db.features_offset, db.features_scale, offset, 6, weight)

def compute_trajectory_direction_feature(db, offset, weight = 1.0):
    for i, feature in enumerate(db.features):
        t0 = db.clamp_trajectory_index(i, 20)
        t1 = db.clamp_trajectory_index(i, 40)
        t2 = db.clamp_trajectory_index(i, 60)

        trajectory_dir0 = quat.mul_vec(quat.inv(db.bone_rotations[i, 0, :]), quat.mul_vec(db.bone_rotations[t0, 0, :], np.array([0, 0, 1], np.float32)))[[0, 2]]
        trajectory_dir1 = quat.mul_vec(quat.inv(db.bone_rotations[i, 0, :]), quat.mul_vec(db.bone_rotations[t1, 0, :], np.array([0, 0, 1], np.float32)))[[0, 2]]
        trajectory_dir2 = quat.mul_vec(quat.inv(db.bone_rotations[i, 0, :]), quat.mul_vec(db.bone_rotations[t2, 0, :], np.array([0, 0, 1], np.float32)))[[0, 2]]


        feature[offset : offset + 6] = np.concatenate([trajectory_dir0, trajectory_dir1, trajectory_dir2])

    normalize_feature(db.features, db.features_offset, db.features_scale, offset, 6, weight)

def build_bounds(db):
    nbound_sm = (db.nframes() + BOUND_SM_SIZE - 1) // BOUND_SM_SIZE
    nbound_lr = (db.nframes() + BOUND_LR_SIZE - 1) // BOUND_LR_SIZE

    db.bound_sm_min = np.full((nbound_sm, db.nfeatures()), np.inf, np.float32)
    db.bound_sm_max = np.full((nbound_sm, db.nfeatures()), -np.inf, np.float32)
    db.bound_lr_min = np.full((nbound_lr, db.nfeatures()), np.inf, np.float32)
    db.bound_lr_max = np.full((nbound_lr, db.nfeatures()), -np.inf, np.float32)

    for i, feature in enumerate(db.features):
        i_sm = i // BOUND_SM_SIZE
        i_lr = i // BOUND_LR_SIZE

        db.bound_sm_min[i_sm] = np.minimum(db.bound_sm_min[i_sm], feature)
        db.bound_sm_max[i_sm] = np.maximum(db.bound_sm_max[i_sm], feature)
        db.bound_lr_min[i_lr] = np.minimum(db.bound_lr_min[i_lr], feature)
        db.bound_lr_max[i_lr] = np.maximum(db.bound_lr_max[i_lr], feature)

def build_matching_features(db,
                            feature_weight_foot_position,
                            feature_weight_foot_velocity,
                            feature_weight_hip_velocity,
                            feature_weight_trajectory_positions,
                            feature_weight_trajectory_directions):
    nfeatures = (
        3 + # left  foot position
        3 + # right foot position
        3 + # left  foot velocity
        3 + # right foot velocity
        3 + # hip velocity
        6 + # trajectory position  2d
        6   # trajectory direction 2d
    )

    db.features = np.zeros((db.nframes(), nfeatures), dtype = np.float32)
    db.features_offset = np.zeros((nfeatures, ))
    db.features_scale  = np.zeros((nfeatures, ))


    compute_bone_position_feature(db, 0, 4, feature_weight_foot_position)
    compute_bone_position_feature(db, 3, 8, feature_weight_foot_position)
    compute_bone_velocity_feature(db, 6, 4, feature_weight_foot_velocity)
    compute_bone_velocity_feature(db, 9, 8, feature_weight_foot_velocity)
    compute_bone_velocity_feature(db, 12, 1, feature_weight_hip_velocity)
    compute_trajectory_position_feature(db, 15, feature_weight_trajectory_positions)
    compute_trajectory_direction_feature(db, 21, feature_weight_trajectory_directions)

    build_bounds(db)

def search(db: Database, query, anim_index, transition_cost = 0.0, ignore_range_end = 20, ignore_surrounding = 20):
    query_normalized = (query - db.features_offset) / db.features_scale
    
    end_of_anim = db.clamp_trajectory_index(anim_index, 1) == anim_index

    if end_of_anim:
        best_index = -1
        best_cost  = float('inf')
    else:
        best_index = anim_index
        best_cost  = np.sum((query_normalized - db.features[anim_index]) ** 2)

    for i, stop in zip(db.range_starts, db.range_stops):
        same_range = i <= anim_index <= stop
        range_end  = stop - ignore_range_end

        while i < range_end:
            i_lr = i // BOUND_LR_SIZE
            i_lr_next = (i_lr + 1) * BOUND_LR_SIZE

            curr_cost = transition_cost + np.sum((query_normalized - clamp(query_normalized, db.bound_lr_min[i_lr], db.bound_lr_max[i_lr])) ** 2)
            if curr_cost >= best_cost:
                i = i_lr_next
                continue

            while i < i_lr_next and i < range_end:
                i_sm = i // BOUND_SM_SIZE
                i_sm_next = (i_sm + 1) * BOUND_SM_SIZE

                curr_cost = transition_cost + np.sum((query_normalized - clamp(query_normalized, db.bound_sm_min[i_sm], db.bound_sm_max[i_sm])) ** 2)
                if curr_cost >= best_cost:
                    i = i_sm_next
                    continue

                while i < i_sm_next and i < range_end:
                    if same_range and (abs(i - anim_index) < ignore_surrounding):
                        i += 1
                        continue
                    
                    curr_cost = transition_cost + np.sum((query_normalized - db.features[i]) ** 2)
                    if curr_cost < best_cost:
                        best_index = i
                        best_cost  = curr_cost

                    i += 1

    return best_index, best_cost