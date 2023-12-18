import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from threading import Thread



transformMatrix = np.eye(4, dtype=np.float32)

def tracking_object():
    window_width = 800
    window_height = 600
    # Transformation matrix

    # Initialize a scaling factor
    scale = 1.0

    panning = False  # Initialize panning flag
    down_x, down_y = 0.0, 0.0
    global delta_x
    global delta_y

    delta_x, delta_y = 0.0, 0.0

    def mouse_button_callback(window, button, action, mods):
        global panning, down_x, down_y
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                panning = True
                down_x, down_y = glfw.get_cursor_pos(window)
                
                print(down_x, down_y)
                #prev_x = prev_x/2
                #prev_y = 1 - 2*prev_y
                
            elif action == glfw.RELEASE:
                panning = False

    
                
    # The scroll callback function
    def scroll_callback(window, x_offset, y_offset):
        global transformMatrix  # Make sure this variable is accessible
        global scale
        scale += y_offset * 0.1  # Adjust this factor as needed
        if scale < 0.1:  # Prevent it from getting too small
            scale = 0.1
            
        transformMatrix[0, 0] = scale  # Scaling in the x-direction
        transformMatrix[1, 1] = scale  # Scaling in the y-direction

    def cursor_position_callback(window, xpos, ypos):
        global transformMatrix  # Make sure this variable is accessible
        # Convert cursor position to OpenGL coordinates if needed
        
        delta_x = xpos - down_x
        delta_y = ypos - down_y
        
        normalized_x = xpos / window_width
        normalized_y = ypos / window_height
        
        opengl_x = 2 * normalized_x - 1
        opengl_y = 1 - 2 * normalized_y
        
        print(normalized_x,normalized_y )


        transformMatrix[0, 3] = opengl_x
        transformMatrix[1, 3] = opengl_y


    # Initialize GLFW
    if not glfw.init():
        raise Exception("GLFW could not be initialized")

    # Create GLFW window
    window = glfw.create_window(window_width, window_height, "My OpenGL Window", None, None)

    # Make the window's OpenGL context current
    glfw.make_context_current(window)

    # Create shader program
    vertex_shader = open("vertex_shader.glsl", "r").read()
    fragment_shader = open("fragment_shader.glsl", "r").read()

    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )

    # Quad vertices
    quad = np.array([
        -0.5, -0.5, 0.0, 0.0, 0.0,
        0.5, -0.5, 0.0, 1.0, 0.0,
        0.5,  0.5, 0.0, 1.0, 1.0,
        -0.5,  0.5, 0.0, 0.0, 1.0
    ], dtype=np.float32)

    # Create VBO and VAO
    VBO = glGenBuffers(1)
    VAO = glGenVertexArrays(1)

    # Bind and upload data
    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, quad, GL_STATIC_DRAW)

    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # Texture coordinates attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    # Generate texture ID
    texture = glGenTextures(1)

    # Bind it as a 2D texture
    glBindTexture(GL_TEXTURE_2D, texture)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    # Create an empty 64x64 texture
    width, height = 64, 64
    texture_data = np.zeros((width, height, 4), dtype=np.uint8)

    # Fill in the pixels
    center_x, center_y = width // 2, height // 2
    radius = min(center_x, center_y)

    for y in range(height):
        for x in range(width):
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx * dx + dy * dy)
            
            if distance < radius:
            #if distance:
                texture_data[y, x] = [255, 255, 255, 255]  # RGBA: White
            else:
                texture_data[y, x] = [0, 0, 0, 0]  # RGBA: Transparent

    # Upload it to GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

    # Unbind the texture
    glBindTexture(GL_TEXTURE_2D, 0)

    glfw.set_cursor_pos_callback(window, cursor_position_callback)

    # Set the scroll callback
    glfw.set_scroll_callback(window, scroll_callback)

    glfw.set_mouse_button_callback(window, mouse_button_callback)

    # Use program
    glUseProgram(shader)

    # Initialize last time checked
    last_time = glfw.get_time()
    frame_count = 0

    transform_loc = glGetUniformLocation(shader, "transformMatrix")
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transformMatrix.T)

    # Loop
    while not glfw.window_should_close(window):
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT)
            
        #if panning:
            # Set the transformation uniform
        transform_loc = glGetUniformLocation(shader, "transformMatrix")
        glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transformMatrix.T)
        
        # In your main loop, before drawing
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)

        texture_loc = glGetUniformLocation(shader, "circleTexture")
        glUniform1i(texture_loc, 0)

        # Draw
        glBindVertexArray(VAO)
        glDrawArrays(GL_QUADS, 0, 4)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        # Calculate FPS
        current_time = glfw.get_time()
        frame_count += 1
        if current_time - last_time >= 1.0:  # If one second has passed
            print(f"FPS: {frame_count}")
            frame_count = 0
            last_time = current_time
        
    glfw.terminate()


if __name__ == "__main__":
    # Run in a separate thread
    #thread = Thread(target=tracking_object)
    #thread.start()
    tracking_object()
