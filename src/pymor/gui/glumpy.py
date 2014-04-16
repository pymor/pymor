# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

''' This module provides a widget for displaying patch plots of
scalar data assigned to a 2D-Grid using OpenGL. This widget is not
intended to be used directly. Instead, use
:meth:`~pymor.gui.qt.visualize_glumpy_patch` or
:class:`~pymor.gui.qt.GlumpyPatchVisualizer`.
'''

from __future__ import absolute_import, division, print_function

import math as m

import numpy as np

try:
    from PySide.QtOpenGL import QGLWidget
    from PySide.QtGui import QSizePolicy, QPainter, QFontMetrics
    HAVE_PYSIDE = True
except ImportError:
    HAVE_PYSIDE = False

try:
    from glumpy.graphics.vertex_buffer import VertexBuffer
    HAVE_GLUMPY = True
except ImportError:
    HAVE_GLUMPY = False

try:
    import OpenGL.GL as gl
    HAVE_GL = True
except ImportError:
    HAVE_GL = False

HAVE_ALL = HAVE_PYSIDE and HAVE_GLUMPY and HAVE_GL


if HAVE_ALL:

    from pymor.grids.constructions import flatten_grid
    from pymor.grids.referenceelements import triangle, square

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

        def __init__(self, parent, grid, vmin=None, vmax=None, bounding_box=[[0, 0], [1, 1]], codim=2):
            assert grid.reference_element in (triangle, square)
            assert grid.dim == 2
            assert codim in (0, 2)
            super(GlumpyPatchWidget, self).__init__(parent)
            self.setMinimumSize(300, 300)
            self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

            subentities, coordinates, entity_map = flatten_grid(grid)

            self.subentities = subentities
            self.entity_map = entity_map
            self.reference_element = grid.reference_element
            self.vmin = vmin
            self.vmax = vmax
            self.bounding_box = bounding_box
            self.codim = codim
            self.update_vbo = False
            bb = self.bounding_box
            self.size = np.array([bb[1][0] - bb[0][0], bb[1][1] - bb[0][1]])
            self.scale = 1 / self.size
            self.shift = - np.array(bb[0]) - self.size / 2

            # setup buffers
            if self.reference_element == triangle:
                if codim == 2:
                    self.vertex_data = np.empty(len(coordinates),
                                                dtype=[('position', 'f4', 4), ('color', 'f4', 4)])
                    self.indices = subentities
                else:
                    self.vertex_data = np.empty(len(subentities) * 3,
                                                dtype=[('position', 'f4', 4), ('color', 'f4', 4)])
                    self.indices = np.arange(len(subentities) * 3, dtype=np.uint32)
            else:
                if codim == 2:
                    self.vertex_data = np.empty(len(coordinates),
                                                dtype=[('position', 'f4', 4), ('color', 'f4', 4)])
                    self.indices = np.vstack((subentities[:, 0:3], subentities[:, [0, 2, 3]]))
                else:
                    self.vertex_data = np.empty(len(subentities) * 6,
                                                dtype=[('position', 'f4', 4), ('color', 'f4', 4)])
                    self.indices = np.arange(len(subentities) * 6, dtype=np.uint32)

            self.vertex_data['position'][:, 2] = 0
            self.vertex_data['position'][:, 3] = 0.5
            self.vertex_data['color'] = 1

            self.set_coordinates(coordinates)
            self.set(np.zeros(grid.size(codim)))

        def resizeGL(self, w, h):
            gl.glViewport(0, 0, w, h)
            gl.glLoadIdentity()
            self.update()

        def initializeGL(self):
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self.shaders_program = link_shader_program(compile_vertex_shader(VS))
            gl.glUseProgram(self.shaders_program)
            self.vbo = VertexBuffer(self.vertex_data, indices=self.indices)

        def paintGL(self):
            if self.update_vbo:
                self.vbo.upload()
                self.update_vbo = False
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            self.vbo.draw(gl.GL_TRIANGLES, 'pc')

        def set_coordinates(self, coordinates):
            if self.codim == 2:
                self.vertex_data['position'][:, 0:2] = coordinates
                self.vertex_data['position'][:, 0:2] += self.shift
                self.vertex_data['position'][:, 0:2] *= self.scale
            elif self.reference_element == triangle:
                VERTEX_POS = coordinates[self.subentities]
                VERTEX_POS += self.shift
                VERTEX_POS *= self.scale
                self.vertex_data['position'][:, 0:2] = VERTEX_POS.reshape((-1, 2))
            else:
                num_entities = len(self.subentities)
                VERTEX_POS = coordinates[self.subentities]
                VERTEX_POS += self.shift
                VERTEX_POS *= self.scale
                self.vertex_data['position'][0:num_entities * 3, 0:2] = VERTEX_POS[:, 0:3, :].reshape((-1, 2))
                self.vertex_data['position'][num_entities * 3:, 0:2] = VERTEX_POS[:, [0, 2, 3], :].reshape((-1, 2))
            self.update_vbo = True
            self.update()

        def set(self, U):
            U_buffer = self.vertex_data['color'][:, 0]
            if self.codim == 2:
                U_buffer[:] = U[self.entity_map]
            elif self.reference_element == triangle:
                U_buffer[:] = np.repeat(U, 3)
            else:
                U_buffer[:] = np.tile(np.repeat(U, 3), 2)

            # normalize
            vmin = np.min(U) if self.vmin is None else self.vmin
            vmax = np.max(U) if self.vmax is None else self.vmax
            U_buffer -= vmin
            if (vmax - vmin) > 0:
                U_buffer /= float(vmax - vmin)

            self.update_vbo = True
            self.update()

    class ColorBarWidget(QGLWidget):

        def __init__(self, parent, U=None, vmin=None, vmax=None):
            super(ColorBarWidget, self).__init__(parent)
            self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding))
            self.setAutoFillBackground(False)
            self.set(U, vmin, vmax)

        def resizeGL(self, w, h):
            gl.glViewport(0, 0, w, h)
            gl.glLoadIdentity()
            self.update()

        def initializeGL(self):
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self.shaders_program = link_shader_program(compile_vertex_shader(VS))
            gl.glUseProgram(self.shaders_program)

        def set(self, U=None, vmin=None, vmax=None):
            # normalize U
            fm = QFontMetrics(self.font())
            self.vmin = vmin if vmin is not None else (np.min(U) if U is not None else 0.)
            self.vmax = vmax if vmax is not None else (np.max(U) if U is not None else 1.)
            precision = m.log(max(abs(self.vmin), abs(self.vmax) / abs(self.vmin - self.vmax)), 10) + 1
            precision = int(min(max(precision, 3), 8))
            self.vmin_str = format(('{:.' + str(precision) + '}').format(self.vmin))
            self.vmax_str = format(('{:.' + str(precision) + '}').format(self.vmax))
            self.vmin_width = fm.width(self.vmin_str)
            self.vmax_width = fm.width(self.vmax_str)
            self.text_height = fm.height() * 1.5
            self.text_ascent = fm.ascent() * 1.5
            self.text_descent = fm.descent() * 1.5
            self.setMinimumSize(max(self.vmin_width, self.vmax_width) + 20, 300)
            self.update()

        def paintGL(self):
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glUseProgram(self.shaders_program)

            gl.glBegin(gl.GL_QUAD_STRIP)
            bar_start = -1 + self.text_height / self.height() * 2
            bar_height = (1 - 2 * self.text_height / self.height()) * 2
            steps = 40
            for i in xrange(steps + 1):
                y = i * (1 / steps)
                gl.glColor(y, 0, 0)
                gl.glVertex(-0.5, (bar_height*y + bar_start), 0.0)
                gl.glVertex(0.5, (bar_height*y + bar_start), 0.0)
            gl.glEnd()
            p = QPainter(self)
            p.drawText((self.width() - self.vmax_width)/2, self.text_ascent, self.vmax_str)
            p.drawText((self.width() - self.vmin_width)/2, self.height() - self.text_height + self.text_ascent,
                       self.vmin_str)
            p.end()

else:

    class GlumpyPatchWidget(object):
        pass

    class ColorBarWidget(object):
        pass
