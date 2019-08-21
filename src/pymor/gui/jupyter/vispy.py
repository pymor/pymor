from vispy import gloo
from vispy import app
from vispy.app.backends.ipython import VispyWidget
from vispy.util.transforms import perspective, translate, rotate
from matplotlib.cm import get_cmap
import IPython
import numpy as np
from ipywidgets import IntSlider, interact, widgets, Play

from pymor.grids.referenceelements import triangle, square
from pymor.grids.constructions import flatten_grid
from pymor.vectorarrays.interfaces import VectorArrayInterface

# we should try to limit ourselver to webgl 1.0 here since 2.0 (draft) is not as widely supported
# https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API#Browser_compatibility
EL_VS = """#version 100 es
    // Attribute variable that contains coordinates of the vertices.
    uniform sampler2D colormap;
    attribute vec2 position;
    attribute float color;
    varying float texcoord;

    void main()
    {
        gl_Position.xy = position.xy;
        gl_Position.z = 0.;
        gl_Position.w = 1.;
        texcoord = color; 
    }
    """

EL_FS = """#version 100 es
    uniform sampler2D colormap;
    varying float texcoord;

    void main()
    {
        gl_FragColor = texture2D(colormap, vec2(texcoord, 0.));
    }
    """


class Canvas(app.Canvas):

    def __init__(self, grid, color_map, title, vmin=None, vmax=None,
                 bounding_box=([0, 0], [1, 1]), codim=2,
                 ):
        app.Canvas.__init__(self, keys='interactive', size=(480, 480), title=title)
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)
        self.ps = self.pixel_scale
        self.grid = grid
        self.color_map = color_map

        self.translate = 40
        self.program = gloo.Program(EL_VS, EL_FS)
        self.view = translate((0, 0, -self.translate))
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.apply_zoom()

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
        self.vsize = np.array([bb[1][0] - bb[0][0], bb[1][1] - bb[0][1]])
        self.scale = 2 / self.vsize
        self.shift = - np.array(bb[0]) - self.vsize / 2

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
        self.indices_data = np.ascontiguousarray(self.indices, dtype=np.uint32)
        self.indices = gloo.IndexBuffer(self.indices_data)

        self.vertex_data['color'] = 1

        self.set_coordinates(coordinates)
        self.set(np.zeros(grid.size(codim)))

        self.initialize_gl()

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

    def set(self, U, vmin=None, vmax=None):
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

        self.update_vbo = True
        self.update()

    def initialize_gl(self):
        gloo.set_state(depth_test=True, clear_color='white')
        cm = self.color_map(np.linspace(0., 1., min(gloo.gl.GL_MAX_TEXTURE_SIZE, 1024))).astype(np.float32)[:, 0:3]
        cm = np.ascontiguousarray(cm[None, :, :])
        self.program['colormap'] = cm
        self.program['colormap'].interpolation = 'linear'

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def apply_zoom(self):
        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)

    # def on_mouse_wheel(self, event):
    #     self.translate -= event.delta[1]
    #     self.translate = max(-1, self.translate)
    #     self.view = translate((0, 0, -self.translate))
    #
    #     self.program['u_view'] = self.view
    #     self.update()

    def on_draw(self, event):
        if self.update_vbo:
            self.program.bind(gloo.VertexBuffer(self.vertex_data))
            self.update_vbo = False
        gloo.clear()
        self.program.draw('triangles', self.indices)


def visualize_vispy(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, columns=2,
         color_map=get_cmap('viridis')):
    """Generate a vispy Plot and associated controls for  scalar data associated to a two-dimensional |Grid|.

    The grid's |ReferenceElement| must be the triangle or square. The data can either
    be attached to the faces or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    U
        |VectorArray| of the data to visualize. If `len(U) 1`, the data is visualized
        as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
        provided, in which case a subplot is created for each entry of the tuple. The
        lengths of all arrays have to agree.
    bounding_box
        A bounding box in which the grid is contained.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 2).
    title
        Title of the plot.
    legend
        Description of the data that is plotted. Most useful if `U` is a tuple in which
        case `legend` has to be a tuple of strings of the same length.
    separate_colorbars
        If `True`, use separate colorbars for each subplot.
    rescale_colorbars
        If `True`, rescale colorbars to data in each frame.
    columns
        The number of columns in the visualizer GUI in case multiple plots are displayed
        at the same time.
    color_map
        a Matplotlib Colormap object
    """
    assert isinstance(U, VectorArrayInterface) \
           or (isinstance(U, tuple) and all(isinstance(u, VectorArrayInterface) for u in U)
               and all(len(u) == len(U[0]) for u in U))
    if isinstance(U, VectorArrayInterface):
        size = len(U)
        U = (U.to_numpy().astype(np.float64, copy=False),)
    else:
        size = len(U[0])
        U = tuple(u.to_numpy().astype(np.float64, copy=False) for u in U)

    if separate_colorbars:
        if rescale_colorbars:
            vmins = tuple(np.min(u[0]) for u in U)
            vmaxs = tuple(np.max(u[0]) for u in U)
        else:
            vmins = tuple(np.min(u) for u in U)
            vmaxs = tuple(np.max(u) for u in U)
    else:
        if rescale_colorbars:
            vmins = (min(np.min(u[0]) for u in U),) * len(U)
            vmaxs = (max(np.max(u[0]) for u in U),) * len(U)
        else:
            vmins = (min(np.min(u) for u in U),) * len(U)
            vmaxs = (max(np.max(u) for u in U),) * len(U)

    vispy_widgets = []
    for u, vmin, vmax in zip(U, vmins, vmaxs):
        c = Canvas(grid=grid, bounding_box=bounding_box, vmin=vmin, vmax=vmax, title=title, color_map=color_map)
        c.set(u[0])
        v = VispyWidget()
        v.set_canvas(c)
        vispy_widgets.append(v)
    canvases = widgets.HBox(vispy_widgets)

    if size > 1:
        def _goto_idx(idx):
            for c, u in zip(vispy_widgets, U):
                c.canvas.set(u[idx])
        play = Play(min=0, max=size - 1, step=1, value=0, description='Timestep:')
        interact(idx=play).widget(_goto_idx)
        slider = IntSlider(min=0, max=size - 1, step=1, value=0, description='Timestep:')
        widgets.jslink((play, 'value'), (slider, 'value'))
        controls = widgets.HBox([play, slider])
        canvases = widgets.VBox([canvases, controls])
    IPython.display.display(canvases)
    return canvases
