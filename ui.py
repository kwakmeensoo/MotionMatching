import imgui
import glfw
from imgui.integrations.glfw import GlfwRenderer
from tkinter import filedialog

class UI:
    def __init__(self, engine, controller):
        self.engine = engine
        self.window = self.engine.window
        self.controller = controller
        
        self.selected_radio = 0
        self.file_path = None

        imgui.create_context()
        self.impl = GlfwRenderer(self.window, False)
        self.focused = False

        self._setup_callbacks()

    def _setup_callbacks(self):
        # glfw에는 하나의 callback 밖에 등록할 수 없으므로,
        # 기존 callback과 통합하여 새로운 callback을 만든다.
        glfw.set_key_callback(self.window, self.custom_key_callback)
        glfw.set_cursor_pos_callback(self.window, self.custom_cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.custom_scroll_callback)

        glfw.set_window_size_callback(self.window, self.impl.resize_callback)
        glfw.set_char_callback(self.window, self.impl.char_callback)

    def custom_key_callback(self, *args):
        self.impl.keyboard_callback(*args)

        if not self.focused:
            self.controller.key_callback(*args)
    
    def custom_cursor_pos_callback(self, *args):
        self.impl.mouse_callback(*args)

        if not self.focused:
            self.controller.cursor_pos_callback(*args)
    
    def custom_scroll_callback(self, *args):
        self.impl.scroll_callback(*args)

        if not self.focused:
            self.controller.scroll_callback(*args)

    def render(self):
        self.impl.process_inputs()
        imgui.new_frame()
        self._render_ui()
        imgui.end_frame()

        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def _render_ui(self):
        imgui.begin('Control Panel', flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        self.focused = imgui.is_window_focused(flags = imgui.FOCUS_CHILD_WINDOWS)

        if imgui.button('Open', width = 190):
            filename = filedialog.askopenfilename(
                title = '파일 선택',
                filetypes = [('BVH Files', '*.bvh')],
                initialdir = '.'
            )

            if len(filename) > 0:
                self.engine.init_bvh(filename)

        with imgui.begin_child('Frame Control Window', border = True, height = 107):
            imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (4, 5))
            imgui.columns(3, border = False)
            if imgui.button('Play', width = imgui.get_content_region_available_width()):
                self.engine.is_playing = True
            imgui.next_column()

            if imgui.button('Stop', width = imgui.get_content_region_available_width()):
                self.engine.is_playing = False
            imgui.next_column()
            
            if imgui.button('Reset', width = imgui.get_content_region_available_width()):
                self.engine.curr_frame = 0
                self.engine.is_playing = False

            imgui.columns(2, border = False)
            if imgui.button('Prev', width = imgui.get_content_region_available_width()):
                self.engine.curr_frame = max(0, self.engine.curr_frame - 1)
            imgui.next_column()

            if imgui.button('Next', width = imgui.get_content_region_available_width()):
                self.engine.curr_frame = min(self.engine.num_frames - 1, self.engine.curr_frame + 1)

            imgui.columns(1, border = False)
            imgui.align_text_to_frame_padding()
            imgui.text('Frame:')
            imgui.same_line()
            imgui.set_next_item_width(imgui.get_content_region_available_width())
            frame_changed, frame_value = imgui.slider_int(
                '##frame', self.engine.curr_frame + 1, 1, self.engine.num_frames
            )

            if frame_changed:
                self.engine.curr_frame = frame_value - 1

            imgui.columns(1, border = False)
            imgui.align_text_to_frame_padding()
            imgui.text('Speed:')
            imgui.same_line()
            imgui.set_next_item_width(imgui.get_content_region_available_width())
            speed_changed, speed_value = imgui.slider_float(
                '##speed', self.engine.speed, 0.25 if self.engine.skeletons else 1.0 , 2.0 if self.engine.skeletons else 1.0, format = '%.2f'
            )

            if speed_changed:
                self.engine.speed = speed_value
                self.engine.update_speed()

            imgui.pop_style_var()
        
        imgui.end()