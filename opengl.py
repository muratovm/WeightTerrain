from OpenGL.GL import *
from OpenGL import GLUT
import glfw
import numpy as np

def draw_text(x, y, text, font=GLUT.GLUT_BITMAP_8_BY_13):
    glRasterPos2f(x, y)
    for ch in text:
        GLUT.glutBitmapCharacter(font, ord(ch))

def draw_symmetry_line():
    glColor3f(0, 1, 0)  # Set color to green (or any color you prefer)
    glBegin(GL_LINES)
    glVertex2f(-1, 0)  # Start point at left side of the window
    glVertex2f(1, 0)   # End point at right side of the window
    glEnd()


