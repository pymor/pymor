from pymor.grids.referenceelements import triangle, square
import time
import math
import numpy as np

try:
    from kivy.graphics import RenderContext
    from kivy.uix.widget import Widget
    from kivy.uix.label import Label
    HAVE_KIVY = True
except ImportError:
    HAVE_KIVY = False

try:
    from kivy.graphics.opengl import *
    HAVE_GL = True
except ImportError:
    HAVE_GL = False

HAVE_ALL = HAVE_KIVY and HAVE_GL

if HAVE_ALL:

    def getGLPatchWidget(parent, grid, bounding_box=None, vmin=None, vmax=None, codim=2, dpi=100):

        from kivy.modules import inspector
        from kivy.core.window import Window
        from kivy.graphics import Mesh
        from kivy.uix.widget import Widget
        from kivy.uix.boxlayout import BoxLayout
        from kivy.core.window import Window
        from kivy.uix.widget import Widget
        from kivy.graphics.transformation import Matrix
        from kivy.resources import resource_find, resource_add_path

        from kivy.graphics import Fbo, Rectangle, Callback

        from pymor.grids.constructions import flatten_grid

        import OpenGL.GL as gl
        import os
        import inspect
        from ctypes import c_void_p

        # add the path to this module to kivys known paths. This allows kivy to find the shader files
        module_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        resource_add_path(module_path)
        print(module_path)

        class GLPatchWidgetFBO(Widget):

            # bound for the number of vertices in one Mesh
            MAX_VERTICES = 2**16//3
            # size of the FBO, it's texture gets resized and mapped to a renctangle in the GUI
            FBO_SIZE = (1000, 1000)

            def __init__(self, grid, vmin=None, vmax=None, bounding_box=([0, 0], [1, 1]), codim=2):
                assert grid.reference_element in (triangle, square)
                assert grid.dim == 2
                assert codim in (0, 2)

                super(GLPatchWidgetFBO, self).__init__()

                self.grid = grid

                subentities, coordinates, entity_map = flatten_grid(grid)

                self.subentities = subentities
                self.entity_map = entity_map
                self.reference_element = grid.reference_element
                self.vmin = vmin
                self.vmax = vmax
                self.bounding_box = bounding_box
                self.codim = codim

                bb = self.bounding_box
                size_ = np.array([bb[1][0] - bb[0][0], bb[1][1] - bb[0][1]])

                self.shift = bb[0]
                self.scale = 1. / size_

                # setup buffers
                if self.reference_element == triangle:
                    if codim == 2:
                        self.vertex_data = np.empty((len(coordinates), 3))
                        self.indices = np.asarray(subentities)
                    else:
                        self.vertex_data = np.empty((len(subentities)*3, 3))
                        self.indices = np.arange(len(subentities) * 3, dtype=np.uint32)
                else:
                    if codim == 2:
                        self.vertex_data = np.empty((len(coordinates), 3))
                        self.indices = np.vstack((subentities[:, 0:3], subentities[:, [0, 2, 3]]))
                    else:
                        self.vertex_data = np.empty((len(subentities)*6, 3))
                        self.indices = np.arange(len(subentities) * 6, dtype=np.uint32)

                self.set_coordinates(coordinates)
                self.meshes = None

                with self.canvas:
                    self.fbo = Fbo(use_parent_modelview=True, size=self.FBO_SIZE)
                    self.rect = Rectangle(texture=self.fbo.texture)

                self.fbo.shader.source = resource_find("shader_mesh.glsl")

                self.bind(pos=self.on_pos)
                self.bind(size=self.on_size)

                # this lets you inspect the user interface with the shortcut Ctrl+E
                # inspector.create_inspector(Window, self)

            # todo optimization (prevent copying of vertices)
            # to do so the vertices for each mesh must be continuous in memory
            def update_meshes(self):
                start = time.time()
                num_meshes = len(self.meshes)

                if num_meshes == 1:
                    self.meshes[0].vertices = self.vertex_data.reshape((-1))
                else:
                    for i in range(num_meshes-1):
                        ind = self.index_map[i]
                        self.meshes[i].vertices = self.vertex_data[ind].reshape((-1))

                    ind = self.index_map[-1]
                    self.meshes[-1].vertices = self.vertex_data[ind].reshape((-1))

                stop = time.time()
                m_str = "mesh" if num_meshes == 1 else "meshes"
                print("Update of {} {} took {} seconds".format(num_meshes, m_str, stop-start))

            def create_meshes(self):

                start = time.time()
                max_vertices = self.MAX_VERTICES

                num_vertices = len(self.indices)
                num_meshes = int(math.ceil(num_vertices/max_vertices))

                print("num_meshes", num_meshes)
                print("num_vertices", num_vertices)
                print("max_vertices", max_vertices)

                vertex_format = [
                    (b'v_pos', 2, 'float'),
                    (b'v_color', 1, 'float'),
                ]

                self.index_map = []

                if num_meshes == 1 or num_vertices < max_vertices*3:
                    # if the number of vertices doesn't exceed max_vertices we can use one mesh
                    ind = self.indices.flatten()
                    self.index_map.append(ind)
                    self.meshes = [Mesh(vertices=self.vertex_data.flatten(), indices=ind, fmt=vertex_format, mode='triangles')]
                else:
                    self.meshes = []
                    for i in range(num_meshes-1):
                        ind = self.indices[i*max_vertices:(i+1)*max_vertices].flatten()
                        self.index_map.append(ind)
                        self.meshes.append(Mesh(vertices=self.vertex_data[ind].flatten(), indices=np.arange(len(ind)),
                                                fmt=vertex_format, mode='triangles'))
                    i = num_meshes - 1
                    ind = self.indices[i*max_vertices:].flatten()
                    self.index_map.append(ind)
                    self.meshes.append(Mesh(vertices=self.vertex_data[ind].flatten(), indices=np.arange(len(ind)),
                                            fmt=vertex_format, mode='triangles'))

                for i, mesh in enumerate(self.meshes):
                    self.fbo.add(mesh)

                end = time.time()

                print("Mesh splitting took {} seconds".format(end-start))

            def set_coordinates(self, coordinates):
                if self.codim == 2:
                    self.vertex_data[:, 0:2] = coordinates
                    self.vertex_data[:, 0:2] += self.shift
                    self.vertex_data[:, 0:2] *= self.scale
                elif self.reference_element == triangle:
                    VERTEX_POS = coordinates[self.subentities]
                    VERTEX_POS += self.shift
                    VERTEX_POS *= self.scale
                    self.vertex_data[:, 0:2] = VERTEX_POS.reshape((-1, 2))
                else:
                    num_entities = len(self.subentities)
                    VERTEX_POS = coordinates[self.subentities]
                    VERTEX_POS += self.shift
                    VERTEX_POS *= self.scale
                    self.vertex_data[0:num_entities * 3, 0:2] = VERTEX_POS[:, 0:3, :].reshape((-1, 2))
                    self.vertex_data[num_entities * 3:, 0:2] = VERTEX_POS[:, [0, 2, 3], :].reshape((-1, 2))

            def set(self, U, vmin=None, vmax=None):
                self.vmin = self.vmin if vmin is None else vmin
                self.vmax = self.vmax if vmax is None else vmax

                U_buffer = self.vertex_data[:, 2]
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

                if self.meshes is None:
                    self.create_meshes()
                else:
                    self.update_meshes()

            def on_pos(self, instance, value):
                self.rect.pos = value

            def on_size(self, instance, value):
                self.rect.texture = self.fbo.texture
                self.rect.size = value

                # the size of the
                self.update_glsl()

            def update_glsl(self, *args):
                w, h = self.size
                w = max(w, 1)
                h = max(h, 1)
                proj = Matrix().view_clip(0, w, 0, h, 1, 100, 0)
                self.fbo['projection_mat'] = proj
                self.fbo['scale'] = [float(v) for v in self.size]


        class GLPatchWidgetClassic(Widget):
            """This class does not work right now, because things with the vertex buffers go totally wrong"""

            # size of the FBO, it's texture gets resized and mapped to a rectangle in the UI
            FBO_SIZE = (1000, 1000)

            def __init__(self, grid, vmin=None, vmax=None, bounding_box=([0, 0], [1, 1]), codim=2):
                assert grid.reference_element in (triangle, square)
                assert grid.dim == 2
                assert codim in (0, 2)

                super(GLPatchWidgetClassic, self).__init__()

                self.grid = grid

                subentities, coordinates, entity_map = flatten_grid(grid)

                self.subentities = subentities
                self.entity_map = entity_map
                self.reference_element = grid.reference_element
                self.vmin = vmin
                self.vmax = vmax
                self.bounding_box = bounding_box
                self.codim = codim

                bb = self.bounding_box
                size_ = np.array([bb[1][0] - bb[0][0], bb[1][1] - bb[0][1]])

                self.shift = bb[0]
                self.scale = 1. / size_

                # setup buffers
                if self.reference_element == triangle:
                    if codim == 2:
                        self.vertex_data = np.empty(len(coordinates),
                                                    dtype=[('position', 'f4', 2), ('color', 'f4', 1)])
                        self.indices = subentities
                    else:
                        self.vertex_data = np.empty(len(subentities) * 3,
                                                    dtype=[('position', 'f4', 2), ('color', 'f4', 1)])
                        self.indices = np.arange(len(subentities) * 3, dtype=np.uint32)
                else:
                    if codim == 2:
                        self.vertex_data = np.empty(len(coordinates),
                                                    dtype=[('position', 'f4', 2), ('color', 'f4', 1)])
                        self.indices = np.vstack((subentities[:, 0:3], subentities[:, [0, 2, 3]]))
                    else:
                        self.vertex_data = np.empty(len(subentities) * 6,
                                                    dtype=[('position', 'f4', 2), ('color', 'f4', 1)])
                        self.indices = np.arange(len(subentities) * 6, dtype=np.uint32)
                self.indices = np.ascontiguousarray(self.indices)

                self.vertex_data['color'] = 1

                #self.set_coordinates(coordinates)

                with self.canvas:
                    self.fbo = Fbo(use_parent_projection=True, use_parent_modelview=True, size=self.FBO_SIZE)
                    self.rect = Rectangle(texture=self.fbo.texture)

                with self.fbo:
                    self.cb = Callback(self.run_gl)

                self.set_coordinates(coordinates)

                self.fbo.shader.source = resource_find("shader_classic.glsl")

                self.bind(pos=self.on_pos)
                self.bind(size=self.on_size)

                # this lets you inspect the user interface with the shortcut Ctrl+E
                inspector.create_inspector(Window, self)

                self.initialized = False

            def run_gl(self, instr):
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

                # use the standard shader
                # if you uncomment this line, the the vertices are positioned correctly, besides taking only the
                # upper right quadrant of the layout. The viewport seems to be [-1, 1]^2 but the vertices are normalized
                # to [0, 1]^2.
                # But the grid is rendered in a solid color as the wrong shader is used.

                # if you comment this line out, the vertex positions seem totally messed up
                #gl.glUseProgram(0)

                if not self.initialized:
                    # set vertices
                    self.vertices_id = gl.glGenBuffers(1)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertices_id)
                    gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertex_data, gl.GL_DYNAMIC_DRAW)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

                    # set indices
                    self.indices_id = gl.glGenBuffers(1)
                    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id)
                    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices, gl.GL_STATIC_DRAW)
                    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

                    self.initialized = True

                if self.need_update:
                    print("UPDATE VERTICES")
                    # set vertices
                    self.vertices_id = gl.glGenBuffers(1)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertices_id)
                    gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertex_data, gl.GL_DYNAMIC_DRAW)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

                    self.need_update = False

                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

                gl.glColor3f(1.0, 1.0, 0.0)

                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertices_id)
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id)

                gl.glVertexPointer(3, gl.GL_FLOAT, 0, c_void_p(None))

                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glDrawElements(gl.GL_TRIANGLES, self.indices.size, gl.GL_UNSIGNED_INT, c_void_p(None))
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

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

                min_x = self.vertex_data['position'][:,0].min()
                max_x = self.vertex_data['position'][:,0].max()
                min_y = self.vertex_data['position'][:,1].min()
                max_y = self.vertex_data['position'][:,1].max()
                print("BB", min_x, max_x, min_y, max_y)

            def set(self, U, vmin=None, vmax=None):
                print("SET")
                self.vmin = self.vmin if vmin is None else vmin
                self.vmax = self.vmax if vmax is None else vmax

                U_buffer = self.vertex_data['color']
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

                self.need_update = True
                self.run_gl(None)

            def on_pos(self, instance, value):
                self.rect.pos = value

            def on_size(self, instance, value):
                self.fbo.size = self.size
                self.rect.texture = self.fbo.texture
                self.rect.size = value

                self.update_glsl()

            def update_glsl(self, *args):
                w, h = self.size
                w = max(w, 1)
                h = max(h, 1)
                proj = Matrix().view_clip(0, w, 0, h, 1, 100, 0)
                self.fbo['projection_mat'] = proj
                self.fbo['scale'] = [float(v) for v in self.size]
                self.fbo['shift'] = [-1.0, -1.0]

        # as stated aove this class doesn't work right now
        """
        return GLPatchWidgetClassic(grid=grid, vmin=vmin, vmax=vmax, bounding_box=bounding_box,
                          codim=codim)
        """
        return GLPatchWidgetFBO(grid=grid, vmin=vmin, vmax=vmax, bounding_box=bounding_box,
                          codim=codim)

    def getColorBarWidget(padding, U=None, vmin=None, vmax=None):

        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.label import Label
        from kivy.graphics.vertex_instructions import Mesh
        from kivy.graphics.transformation import Matrix
        from kivy.graphics import Fbo, Rectangle
        from kivy.resources import resource_find, resource_add_path
        import os
        import inspect

        module_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        resource_add_path(module_path)
        print(module_path)

        class ColorBarFBO(Widget):

            RESOLUTION = 10
            FBO_SIZE = (100, 100)
            BAR_WIDTH = 40

            def __init__(self):

                super(ColorBarFBO, self).__init__()

                with self.canvas:
                    self.fbo = Fbo(use_parent_modelview=True, size=self.FBO_SIZE)
                    self.rect = Rectangle(texture=self.fbo.texture)

                self.fbo.shader.source = resource_find("shader_mesh.glsl")

                self.bind(pos=self.on_pos)
                self.bind(size=self.on_size)

                self.init_mesh()


            def init_mesh(self):
                x = np.array([0.0, 1.0])
                y = np.linspace(0.0, 1.0, self.RESOLUTION)
                vertices = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x)), np.repeat(y, len(x))])

                vertices = vertices.flatten()

                i = np.arange(self.RESOLUTION) * 2
                indices = np.transpose([i, i+1, i+3, i, i+2, i+3]).flatten()

                vertex_format = [
                    (b'v_pos', 2, 'float'),
                    (b'v_color', 1, 'float'),
                ]

                mesh = Mesh(vertices=vertices, indices=indices, fmt=vertex_format, mode='triangles')

                self.fbo.add(mesh)

            def on_pos(self, instance, value):
                x, y = value
                self.rect.pos = [self.center_x - self.BAR_WIDTH//2, y]

            def on_size(self, instance, value):
                width, height = value
                self.rect.size = [self.BAR_WIDTH, height]

                self.update_glsl()

            def update_glsl(self, *args):
                w, h = self.size
                w = max(w, 1)
                h = max(h, 1)
                proj = Matrix().view_clip(0, w, 0, h, 1, 100, 0)
                self.fbo['projection_mat'] = proj
                self.fbo['scale'] = [float(v) for v in self.size]

        class ColorBarWidget(BoxLayout):

            WIDTH = 80
            LABEL_HEIGHT = 40
            LABEL_COLOR = (0, 0, 0, 1)  # RGBA format

            def __init__(self, padding, U=None, vmin=None, vmax=None):
                super(ColorBarWidget, self).__init__(padding=padding, size_hint_x=None, width=self.WIDTH)
                self.label_min = Label(color=self.LABEL_COLOR, size_hint_y=None, height=self.LABEL_HEIGHT)
                self.label_max = Label(color=self.LABEL_COLOR, size_hint_y=None, height=self.LABEL_HEIGHT)
                self.colorbar = ColorBarFBO()

                super(ColorBarWidget, self).__init__(orientation='vertical')
                self.add_widget(self.label_max)
                self.add_widget(self.colorbar)
                self.add_widget(self.label_min)
                self.set(U, vmin, vmax)

            def build(self):
                return self

            def set(self, U=None, vmin=None, vmax=None):
                self.vmin = vmin if vmin is not None else (np.min(U) if U is not None else 0.0)
                self.vmax = vmax if vmax is not None else (np.max(U) if U is not None else 1.0)

                difference = abs(self.vmax - self.vmin)
                if difference == 0:
                    precision = 3
                else:
                    precision = math.log(max(abs(self.vmin), abs(self.vmax)) / difference, 10)
                    precision = int(min(max(precision, 3), 8))
                vmin_str = format(('{:.' + str(precision) + '}').format(self.vmin))
                vmax_str = format(('{:.' + str(precision) + '}').format(self.vmax))

                self.label_max.text = vmax_str
                self.label_min.text = vmin_str

        return ColorBarWidget(padding=padding, U=U, vmin=vmin, vmax=vmax)


else:
    def getGLPatchWidget(parent, grid, bounding_box=None, vmin=None, vmax=None, codim=2, dpi=100):
        return None

    def getColorBarWidget(padding, U=None, vmin=None, vmax=None):
        return None
