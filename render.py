import glfw
import moderngl as mgl
from camera import Camera, CameraMode
from control import Controller
from engine import Engine
from ui import UI

class Renderer():
    def __init__(self, width, height, title):
        self._init_glfw(width, height, title)
        self._init_mgl()

        self.background_color = (0.9, 0.9, 0.9)

        self.camera = Camera(
            position = (5, 4, 5),
            target = (0, 0, 0),
            up = (0, 1, 0),
            aspect = width / height,
            mode = CameraMode.ORBIT
        )

        self.frame_time = 1 / 30

        self.controller = Controller(self.window, self.camera)
        
        self.engine = Engine(self)

        self.ui = UI(self.engine, self.controller)

    def _init_glfw(self, width, height, title, resizable = True, centered = True):
        if not glfw.init():
            raise RuntimeError('GLFW 초기화 실패')
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE if resizable else glfw.FALSE)

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('윈도우 생성 실패')

        if centered:
            monitor = glfw.get_primary_monitor()
            m_xpos, m_ypos = glfw.get_monitor_pos(monitor)
            mode = glfw.get_video_mode(monitor)
            xpos = m_xpos + (mode.size.width - width) // 2
            ypos = m_ypos + (mode.size.height - height) // 2

            glfw.set_window_pos(self.window, xpos, ypos)
        
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)
        glfw.swap_interval(1)

    def _init_mgl(self):
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)

        # # For alpha blending
        # ctx.enable(mgl.BLEND)
        # ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
    
    def _framebuffer_size_callback(self, window, width, height):
        self.ctx.viewport = (0, 0, width, height)
        self.camera.aspect = width / height

    def set_frame_time(self, frame_time):
        self.frame_time = frame_time

    def run(self):
        try:
            lag = 0.0
            prev_time = glfw.get_time()
            while not glfw.window_should_close(self.window):
                curr_time = glfw.get_time()
                elapsed = curr_time - prev_time
                prev_time = curr_time

                lag = lag + elapsed

                glfw.poll_events()
                self.controller.process_input(elapsed)

                while lag >= self.frame_time:
                    self.engine.update()
                    lag -= self.frame_time
                
                self.ctx.clear(*self.background_color)
                self.engine.render()
                self.ui.render()
                glfw.swap_buffers(self.window)
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.engine.cleanup()
        if hasattr(self, 'ctx') and self.ctx:
            self.ctx.release()
        glfw.terminate()