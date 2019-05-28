import random
import ctypes
import math
import os
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import assimp
import pyassimp
import assimp.postprocess
import cyglfw3 as glfw
import numpy
import pygame
import glm
import ctypes
import pywavefront
from pyassimp import *
from assimp import *


# SERGIO MARCHENA
# 16387
# PROYECTO 2
# GRAFICAS POR COMPUTADORA

# config de glfw
glfw.Init()
glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.WindowHint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
glfw.WindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

window=glfw.CreateWindow(800, 600, 'Sergio Marchena - Proyecto 2 - Graficas por Computadora')
glfw.MakeContextCurrent(window)
clock = pygame.time.Clock()
gl.glEnable(gl.GL_DEPTH_TEST)

# VERTEX SHADER (v330)
vertex_shader="""
#version 330
layout (location = 0) in vec4 vert_pos;
layout (location = 1) in vec4 vert_normal;
layout (location = 2) in vec2 vert_texture_coords;
uniform mat4 model_mat;
uniform mat4 view_mat;
uniform mat4 projection_mat;
uniform vec4 color;
uniform vec4 light;
out vec4 vertex_color;
out vec2 vertex_tex_coords;
void main()
{
    float intensity = dot(vert_normal, normalize(light - vert_pos));
    gl_Position = projection_mat * view_mat * model_mat * vert_pos;
    vertex_color = color * intensity;
    vertex_tex_coords = vert_texture_coords;
}
"""
#FRAGMENT SHADER (v330)
fragment_shader_0="""
#version 330
layout (location = 0) out vec4 diffuse_color;
in vec4 vertex_color;
in vec2 vertex_tex_coords;
uniform sampler2D tex;
void main()
{
    diffuse_color = vertex_color * texture(tex, vertex_tex_coords);
}
"""

# vertex_data=numpy.array(
#     [
#         0.1,  0.2,  0.3, 0, 0, 0, 0, 0,
#         0.3,  0.2,  0.1, 0, 0, 0, 0, 0,
#        -0.3, -0.2, -0.1, 0, 0, 0, 0, 0,
#        -0.1, -0.2, -0.3, 0, 0, 0, 0, 0
#     ], dtype=numpy.float32
# )
vertex_data=numpy.array([
    0.5,  0.5, 0, 1, 0, 0, 1, 1,
    0.5, -0.5, 0, 0, 1, 0, 1, 0,
   -0.5, -0.5, 0, 0, 0, 1, 0, 0,
   -0.5,  0.5, 0, 1, 1, 0, 0, 1
], dtype=numpy.float32)

index_data=numpy.array([
    0, 1, 3,
    1, 2, 3,
], dtype=numpy.float32)

vertex_array_object=gl.glGenVertexArrays(1)
gl.glBindVertexArray(vertex_array_object)

vertex_buffer_object=gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer_object)
gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, gl.GL_STATIC_DRAW)

element_buffer_object=gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, gl.GL_STATIC_DRAW)

# glfw requires shaders to be compiled after buffer binding
shader=shaders.compileProgram(shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER), shaders.compileShader(fragment_shader_0, gl.GL_FRAGMENT_SHADER))
gl.glUseProgram(shader)

# MODEL MATRIX
model_mat=glm.mat4(1)
# VIEW MATRIX
view_mat=glm.mat4(1)
# PROJECTION MATRIX
proj_mat=glm.perspective(glm.radians(45), 800 / 600, 0.1, 1000.0)

# IN ORDER TO WORK (in my pc), YOU NEED CHANGE THIS PATH TO THE FULL PATH OF THIS IN THIS PC.
fullpath = "/Users/SergioMarchena/Desktop/2019/GR/GRAFICAS-P2/Proyecto2_Graficas/Japanese_Temple.obj"
texture_name="/Users/SergioMarchena/Desktop/2019/GR/GRAFICAS-P2/Proyecto2_Graficas/textureJ.png"

print("Importing file ...")
newscene = pyassimp.load(fullpath, processing=pyassimp.postprocess.aiProcess_Triangulate)
print("Importing textures ...")
# texture_surface=pygame.image.load(texture_name)
# texture_data=pygame.image.tostring(texture_surface, "RGB", 1)
# width=texture_surface.get_width()
# height=texture_surface.get_height()


# GLIZE
def glize(node):
    model=node.transformation.astype(numpy.float32)

    for mesh in node.meshes:
        width, height=0, 0
        material=dict(mesh.material.properties.items())
        # print(material)

        texture_surface=pygame.image.load(texture_name)
        texture_data=pygame.image.tostring(texture_surface, "RGB", 1)
        width=texture_surface.get_width()
        height=texture_surface.get_height()

        texture=gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, texture_data)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        vertex_data=numpy.hstack((numpy.array(mesh.vertices, dtype=numpy.float32), numpy.array(mesh.normals, dtype=numpy.float32),numpy.array(mesh.texturecoords[0], dtype=numpy.float32)))
        index_data=numpy.hstack(numpy.array(mesh.faces, dtype=numpy.int32))

        vertex_buffer_object=gl.glGenVertexArrays(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer_object)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, gl.GL_STATIC_DRAW)

        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 9 * 4, None)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, 9 * 4, ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, False, 9 * 4, ctypes.c_void_p(6 * 4))
        gl.glEnableVertexAttribArray(2)

        element_buffer_object=gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, gl.GL_STATIC_DRAW)

        shader=shaders.compileProgram(shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER), shaders.compileShader(fragment_shader_0, gl.GL_FRAGMENT_SHADER))
        gl.glUseProgram(shader)

        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "model_mat"), 1, gl.GL_FALSE,model)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "view_mat"), 1, gl.GL_FALSE,glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "projection_mat"), 1, gl.GL_FALSE,glm.value_ptr(proj_mat))

        diffuse=mesh.material.properties["diffuse"]

        gl.glUniform4f(gl.glGetUniformLocation(shader, "color"),*diffuse, 1)
        # LIGHT
        gl.glUniform4f( gl.glGetUniformLocation(shader, "light"),0, 100, 100, 1)
        gl.glDrawElements(gl.GL_TRIANGLES, len(index_data), gl.GL_UNSIGNED_INT, None)

    for child in node.children:
        glize(child)

# CAMERA ATTRIBUTES
cam_radius = 50
cam_angle_xz = 0
cam_angle_xy = 0
camera=glm.vec3(2, 1.2, 80) # FOV

# BACKGROUND COLOR CHANGER
gl.glClearColor(255,255,0,1)


# CONTROLS FOR CAMERA (ARROWS, A,S,ESC)
def key_event(window, key, scancode, action, mods):
    if action == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            glfw.SetWindowShouldClose(window, True)
            print("ESC")
        elif key == glfw.KEY_A:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            print("A _FILL")
        elif key == glfw.KEY_S:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            print("S_LINE")
    elif action == glfw.REPEAT:
        global cam_angle_xz
        global cam_angle_xy
        if key == glfw.KEY_RIGHT:
            cam_angle_xz-=0.4
            print("RIGHT")
        elif key == glfw.KEY_LEFT:
            cam_angle_xz+=0.4
            print("LEFT")
        elif key == glfw.KEY_UP:
            cam_angle_xy+=0.4
            print("UP")
        elif key == glfw.KEY_DOWN:
            cam_angle_xy-=0.4
            print("DOWN")
    camera.x=cam_radius * (math.cos(cam_angle_xz))
    camera.y=cam_radius * (math.sin(cam_angle_xy))
    camera.z=cam_radius * (math.sin(cam_angle_xz))


glfw.SetKeyCallback(window, key_event)
print(camera.x,camera.y,camera.z)

while not glfw.WindowShouldClose(window):
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    #LOOK AT
    view=glm.lookAt(camera, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    glize(newscene.rootnode)
    glfw.SwapBuffers(window)
    clock.tick(5)
    glfw.WaitEvents()

glfw.Terminate()
