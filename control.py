import glfw
import numpy as np
from camera import Camera, CameraMode

class Controller:
    def __init__(self, window, camera: Camera):
        self.window = window
        self.camera = camera

        self.last_x = 0
        self.last_y = 0
        self.first_mouse = True
        self.mouse_pressed = False

        self.keys = {}

        self._setup_callbacks()
    
    def _setup_callbacks(self):
        # 키보드 입력
        glfw.set_key_callback(self.window, self.key_callback)
        # 마우스 클릭
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        # 마우스 이동
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        # 마우스 스크롤
        glfw.set_scroll_callback(self.window, self.scroll_callback)
    
    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.keys[key] = True
        elif action == glfw.RELEASE:
            self.keys[key] = False
        
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        
        if key == glfw.KEY_R and action == glfw.PRESS:
            self.camera.reset()
        
        if key == glfw.KEY_TAB and action == glfw.PRESS:
            if self.camera.mode == CameraMode.FREE:
                self.camera.set_mode(CameraMode.ORBIT)
            else:
                self.camera.set_mode(CameraMode.FREE)
    
    def mouse_button_callback(self, window, button, action, mods):
        if self.camera.mode == CameraMode.FREE:
            if button == glfw.MOUSE_BUTTON_LEFT:
                if action == glfw.PRESS:
                    self.mouse_pressed = True
                    
                elif action == glfw.RELEASE:
                    self.mouse_pressed = False
                    self.first_mouse = True
    
    def cursor_pos_callback(self, window, xpos, ypos):
        if not self.mouse_pressed or self.camera.mode != CameraMode.FREE:
            return
        
        # 왼쪽 마우스 드래그 처리
        # FREE 모드에서 카메라 회전

        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        x_offset = xpos - self.last_x
        y_offset = ypos - self.last_y

        self.last_x = xpos
        self.last_y = ypos

        self.camera.free_rotate(x_offset, y_offset)
    
    def scroll_callback(self, window, x_offset, y_offset):
        self.camera.change_fov(y_offset)
    
    def process_input(self, dt):
        if self.camera.mode == CameraMode.FREE:
            if self.is_key_pressed(glfw.KEY_W):
                self.camera.free_move('forward', dt)
            if self.is_key_pressed(glfw.KEY_S):
                self.camera.free_move('backward', dt)
            if self.is_key_pressed(glfw.KEY_Q):
                self.camera.free_move('up', dt)
            if self.is_key_pressed(glfw.KEY_E):
                self.camera.free_move('down', dt)
            if self.is_key_pressed(glfw.KEY_A):
                self.camera.free_move('left', dt)
            if self.is_key_pressed(glfw.KEY_D):
                self.camera.free_move('right', dt)
        
        if self.camera.mode == CameraMode.ORBIT:
            if self.is_key_pressed(glfw.KEY_A):
                self.camera.orbit_horizontal(dt, 1)
            if self.is_key_pressed(glfw.KEY_D):
                self.camera.orbit_horizontal(dt, -1)
            
            if self.is_key_pressed(glfw.KEY_W):
                self.camera.orbit_vertical(dt, 1)
            if self.is_key_pressed(glfw.KEY_S):
                self.camera.orbit_vertical(dt, -1)
            
            if self.is_key_pressed(glfw.KEY_Q):
                self.camera.orbit_zoom(dt, 1)
            if self.is_key_pressed(glfw.KEY_E):
                self.camera.orbit_zoom(dt, -1)

    def is_key_pressed(self, key):
        return key in self.keys and self.keys[key]