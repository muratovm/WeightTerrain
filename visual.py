import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from threading import Thread
import numpy as np

from opengl import *
from visual_networks import *

# Global variables to track mouse state and position
is_dragging = False
last_pos = (0, 0)

def mouse_button_callback(window, button, action, mods):
    global is_dragging, last_pos
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            is_dragging = True
            last_pos = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            is_dragging = False

def cursor_position_callback(window, xpos, ypos):
    global is_dragging, last_pos
    if is_dragging:
        # Calculate the difference in position
        dx = xpos - last_pos[0]
        dy = ypos - last_pos[1]
        last_pos = (xpos, ypos)

        # Apply the translation to the view
        glTranslatef(dx * 0.0025, -dy * 0.0025, 0)  # Adjust the factor to control the speed of panning

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
    # Set the callback functions
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)

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



