import numpy as np
import glm
from enum import Enum

class CameraMode(Enum):
    FREE = 1    # Free Camera
    ORBIT = 2   # Orbit Camera

class Camera:
    def __init__(self, mode = CameraMode.FREE, position = (0, 5, 5), target = (0, 0, 0), up = (0, 1, 0), fov = 45.0, aspect = 16 / 9, near = 0.1, far = 1000.0):
        self.mode = mode

        self.position = glm.vec3(position)
        self.target = glm.vec3(target)
        self.world_up = glm.vec3(up)
        self.up = glm.vec3(up)

        if mode == CameraMode.ORBIT:
            self._calculate_orbit_params()
        
        self.speed = 5.0
        self.sensitivity = 0.1 # sensitivity for free rotate

        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

        self._update_camera_vectors()
    
    def _calculate_orbit_params(self):
        direction = self.position - self.target

        self.distance = glm.length(direction)

        # xz 평면 상에서의 각도 (경도)
        self.azimuth = np.arctan2(direction.z, direction.x)

        # y축 과의 각도 (위도)
        self.elevation = np.arcsin(direction.y / self.distance)
    
    def _update_from_orbit_params(self):
        x = self.distance * np.cos(self.elevation) * np.cos(self.azimuth)
        y = self.distance * np.sin(self.elevation)
        z = self.distance * np.cos(self.elevation) * np.sin(self.azimuth)

        self.position = self.target + glm.vec3(x, y, z)
    
    def _update_camera_vectors(self):
        # position, target을 바탕으로 front, right, up 계산
        self.front = glm.normalize(self.target - self.position)
        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))
    
    def get_view_matrix(self):
        return glm.lookAt(self.position, self.target, self.up)

    def get_projection_matrix(self):
        return glm.perspective(glm.radians(self.fov), self.aspect, self.near, self.far)
    
    def set_position(self, position):
        if self.mode == CameraMode.Free:
            self.position = glm.vec3(position)
        
        self._update_camera_vectors()
    
    def set_target(self, target):
        if self.mode == CameraMode.FREE:
            return
        
        self.target = glm.vec3(target)

        # target을 따라 다니도록
        self._update_from_orbit_params()
        
        self._update_camera_vectors()
    
    def set_mode(self, mode):
        if self.mode != mode:
            self.mode = mode
        
            if mode == CameraMode.ORBIT:
                self._calculate_orbit_params()
    
    def free_move(self, direction, dt):
        if self.mode == CameraMode.ORBIT:
            return
        displacement = self.speed * dt
        if direction == 'forward':
            self.position += self.front * displacement
            self.target += self.front * displacement
        elif direction == 'backward':
            self.position -= self.front * displacement
            self.target -= self.front * displacement
        elif direction == 'up':
            self.position += self.world_up * displacement
            self.target += self.world_up * displacement
        elif direction == 'down':
            self.position -= self.world_up * displacement
            self.target -= self.world_up * displacement
        elif direction == 'left':
            self.position -= self.right * displacement
            self.target -= self.right * displacement
        elif direction == 'right':
            self.position += self.right * displacement
            self.target += self.right * displacement
        self._update_camera_vectors()
    
    def free_rotate(self, x_offset, y_offset):
        if self.mode == CameraMode.ORBIT:
            return
        x_offset *= self.sensitivity
        y_offset *= self.sensitivity

        rotated_front = self.target - self.position

        # y_offset은 마우스가 아래로 갈수록 양수
        rotation_y = glm.rotate(glm.mat4(1.0), glm.radians(-y_offset), self.right)
        rotated_front = glm.vec3(rotation_y * glm.vec4(rotated_front, 0.0))

        rotation_x = glm.rotate(glm.mat4(1.0), glm.radians(-x_offset), self.world_up)
        rotated_front = glm.vec3(rotation_x * glm.vec4(rotated_front, 0.0))

        self.target = self.position + rotated_front

        self._update_camera_vectors()

    def orbit_horizontal(self, dt, direction = 1):
        if self.mode == CameraMode.FREE:
            return
        self.azimuth += direction * self.speed * dt * 0.5

        self._update_from_orbit_params()
        self._update_camera_vectors()

    def orbit_vertical(self, dt, direction = 1):
        if self.mode == CameraMode.FREE:
            return
        self.elevation += direction * self.speed * dt * 0.5
        if self.elevation > np.pi / 2 - 0.1:
            self.elevation = np.pi / 2 - 0.1
        if self.elevation < -np.pi / 2 + 0.1:
            self.elevation = -np.pi / 2 + 0.1
        
        self._update_from_orbit_params()
        self._update_camera_vectors()
    
    def orbit_zoom(self, dt, direction = 1):
        if self.mode == CameraMode.ORBIT:
            self.distance -= direction * dt * self.speed * 0.5

            if self.distance < 0.15:
                self.distance = 0.15
        
        self._update_from_orbit_params()
        self._update_camera_vectors()

    # mouse scroll event
    def change_fov(self, y_offset):
        self.fov -= y_offset
        if self.fov < 10.0:
            self.fov = 10.0
        if self.fov > 90.0:
            self.fov = 90.0
    
    def reset(self, mode = None, position = (0, 5, 5), target = (0, 0, 0)):
        if mode is not None:
            self.mode = mode
        
        self.position = glm.vec3(position)
        self.target = glm.vec3(target)

        if self.mode == CameraMode.ORBIT:
            self._calculate_orbit_params()
        
        self._update_camera_vectors()