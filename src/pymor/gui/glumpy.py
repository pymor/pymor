# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from PySide.QtOpenGL import QGLWidget
from PySide.QtGui import QSizePolicy
from glumpy.graphics.vertex_buffer import VertexBuffer
import OpenGL.GL as gl

from pymor.la.numpyvectorarray import NumpyVectorArray
from pymor.grids.referenceelements import triangle


def compile_vertex_shader(source):
    """Compile a vertex shader from source."""
    vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vertex_shader, source)
    gl.glCompileShader(vertex_shader)
    # check compilation error
    result = gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetShaderInfoLog(vertex_shader))
    return vertex_shader


def link_shader_program(vertex_shader):
    """Create a shader program with from compiled shaders."""
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glLinkProgram(program)
    # check linking error
    result = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetProgramInfoLog(program))
    return program

VS = """
#version 120
// Attribute variable that contains coordinates of the vertices.
attribute vec4 position;

vec3 getJetColor(float value) {
     float fourValue = 4 * value;
     float red   = min(fourValue - 1.5, -fourValue + 4.5);
     float green = min(fourValue - 0.5, -fourValue + 3.5);
     float blue  = min(fourValue + 0.5, -fourValue + 2.5);

     return clamp( vec3(red, green, blue), 0.0, 1.0 );
}
void main()
{
    gl_Position = position;
    gl_FrontColor = vec4(getJetColor(gl_Color.x), 1);
}
"""

class GlumpyPatchWidget(QGLWidget):

    def __init__(self, parent, grid, vmin=None, vmax=None):
        assert grid.reference_element == triangle
        super(GlumpyPatchWidget, self).__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.grid = grid
        self.U = np.zeros(grid.size(2))
        self.vmin = vmin
        self.vmax = vmax

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)
        gl.glLoadIdentity()
        self.update()

    def upload_buffer(self):
        self.vbo.vertices['color'] = np.hstack((self.U[..., np.newaxis].astype('f4'),
                                                np.zeros((self.U.size, 2), dtype='f4'),
                                                np.ones((self.U.size, 1), dtype='f4')))
        self.vbo.upload()

    def initializeGL(self):
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glShadeModel(gl.GL_SMOOTH)
        self.shaders_program = link_shader_program(compile_vertex_shader(VS))
        g = self.grid
        x, y = g.centers(2)[:, 0] - 0.5, g.centers(2)[:, 1] - 0.5
        lpos = np.array([(x[i], y[i], 0, 0.5) for i in xrange(g.size(2))],
                        dtype='f')
        vertex_data = np.array([(lpos[i], (1, 1, 1, 1)) for i in xrange(g.size(2))],
                               dtype=[('position', 'f4', 4), ('color', 'f4', 4)])
        self.vbo = VertexBuffer(vertex_data, indices=g.subentities(0, 2))
        gl.glUseProgram(self.shaders_program)
        self.set(self.U)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.vbo.draw(gl.GL_TRIANGLES, 'pc')

    def set(self, U):
        # normalize U
        U = np.array(U)
        vmin = np.min(U) if self.vmin is None else self.vmin
        vmax = np.max(U) if self.vmax is None else self.vmax
        U -= vmin
        U /= float(vmax - vmin)
        self.U = U
        if hasattr(self, 'vbo'):
            self.upload_buffer()
        self.update()
