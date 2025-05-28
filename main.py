from render import Renderer
import tkinter as tk

if __name__ == '__main__':
    width = 1280
    height = 720

    # for filedialog
    tk_app = tk.Tk()
    tk_app.withdraw()

    renderer = Renderer(width, height, "BVH Player")

    renderer.run()