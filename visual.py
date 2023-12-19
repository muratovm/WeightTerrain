import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from threading import Thread
import numpy as np

from opengl import *
from visual_networks import *

def run_glfw_window(network):
    if not glfw.init():
        return
    window_width = 800
    window_height = 600
    window = glfw.create_window(window_width, window_height, "GLFW Window", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw network here
        draw_symmetry_line()
        network.draw()

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    # Run in a separate thread
    network = Network([15, 5,20,15,15,5,10])  # Example: 3 layers with 3, 4, and 2 neurons respectively
    thread = Thread(target=run_glfw_window, args=(network,))
    thread.start()



