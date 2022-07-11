# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
import asyncio
from io import BytesIO

import numpy as np
from ipywidgets import IntSlider, interact, widgets, Play, Layout, Label
import pythreejs as p3js
from matplotlib.cm import get_cmap

from pymor.core import config
from pymor.discretizers.builtin.grids.referenceelements import triangle, square
from pymor.discretizers.builtin.grids.constructions import flatten_grid
from pymor.vectorarrays.interface import VectorArray

# we should try to limit ourselves to webgl 1.0 here since 2.0 (draft) is not as widely supported
# https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API#Browser_compatibility
# version directives and such are preprended by threejs
RENDER_VERTEX_SHADER = """
    attribute float data;
    varying float texcoord;

    void main()
    {
        texcoord = data;
        gl_Position =  projectionMatrix * modelViewMatrix * vec4(position,1.0);
    }
    """

RENDER_FRAGMENT_SHADER = """
    uniform sampler2D colormap;
    varying float texcoord;

    void main()
    {
        gl_FragColor = texture2D(colormap, vec2(texcoord, 0.));
    }
    """


def _normalize(u, vmin=None, vmax=None):
    # rescale to be in [max(0,vmin), min(1,vmax)], scale nan to be the smallest value
    vmin = np.nanmin(u) if vmin is None else vmin
    vmax = np.nanmax(u) if vmax is None else vmax
    u -= vmin
    if (vmax - vmin) > 0:
        u /= float(vmax - vmin)
    return np.nan_to_num(u)


class Renderer(widgets.VBox):
    def __init__(self, U, grid, render_size, color_map, title, bounding_box=([0, 0], [1, 1]), codim=2,
                 vmin=None, vmax=None):
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)
        self.layout = Layout(min_width=str(render_size[0]), min_height=str(render_size[1]), margin='10px')
        self.grid = grid
        self.codim = codim
        self.vmin, self.vmax = vmin, vmax

        subentities, coordinates, self.entity_map = flatten_grid(grid)

        if grid.reference_element == triangle:
            if codim == 2:
                vertices = np.zeros((len(coordinates), 3))
                vertices[:, :-1] = coordinates
                indices = subentities
            else:
                vertices = np.zeros((len(subentities) * 3, 3))
                VERTEX_POS = coordinates[subentities]
                vertices[:, 0:2] = VERTEX_POS.reshape((-1, 2))
                indices = np.arange(len(subentities) * 3, dtype=np.uint32)
        else:
            if codim == 2:
                vertices = np.zeros((len(coordinates), 3))
                vertices[:, :-1] = coordinates
                indices = np.vstack((subentities[:, 0:3], subentities[:, [0, 2, 3]]))
            else:
                num_entities = len(subentities)
                vertices = np.zeros((num_entities * 6, 3))
                VERTEX_POS = coordinates[subentities]
                vertices[0:num_entities * 3, 0:2] = VERTEX_POS[:, 0:3, :].reshape((-1, 2))
                vertices[num_entities * 3:, 0:2] = VERTEX_POS[:, [0, 2, 3], :].reshape((-1, 2))
                indices = np.arange(len(subentities) * 6, dtype=np.uint32)

        max_tex_size = 512
        cm = color_map(np.linspace(0, 1, max_tex_size)).astype(np.float32)
        cm.resize((max_tex_size, 1, 4))
        color_map = p3js.DataTexture(cm, format='RGBAFormat',  width=max_tex_size, height=1, type='FloatType')
        uniforms = dict(
            colormap={'value': color_map, 'type': 'sampler2D'},
        )
        self.material = p3js.ShaderMaterial(vertexShader=RENDER_VERTEX_SHADER, fragmentShader=RENDER_FRAGMENT_SHADER,
                                            uniforms=uniforms)

        self.buffer_vertices = p3js.BufferAttribute(vertices.astype(np.float32), normalized=False)
        self.buffer_faces    = p3js.BufferAttribute(indices.astype(np.uint32).ravel(), normalized=False)
        self.meshes = []
        self._setup_scene(bounding_box, render_size)
        if config.is_nbconvert():
            # need to ensure all data is loaded before cell execution is over
            class LoadDummy:
                def done(self):
                    return True
            self._load_data(U)
            self.load = LoadDummy()
        else:
            self.load = asyncio.ensure_future(self._async_load_data(U))

        self._last_idx = None
        super().__init__(children=[self.renderer, ])

    def _get_mesh(self, i, u):
        if self.codim == 2:
            u = u[self.entity_map]
        elif self.grid.reference_element == triangle:
            u = np.repeat(u, 3)
        else:
            u = np.tile(np.repeat(u, 3), 2)
        data = p3js.BufferAttribute(_normalize(u, self.vmin[i], self.vmax[i]), normalized=True)
        geo = p3js.BufferGeometry(
            index=self.buffer_faces,
            attributes=dict(
                position=self.buffer_vertices,
                data=data
            )
        )
        mesh = p3js.Mesh(geometry=geo, material=self.material)
        mesh.visible = False
        # translate to origin where the camera is looking by default,
        # avoids camera not updating in nbconvert run
        mesh.position = tuple(p-i for p, i in zip(mesh.position, self.mesh_center))
        return mesh

    async def _async_load_data(self, data):
        self._load_data(data)

    def _load_data(self, data):
        for i, u in enumerate(data):
            m = self._get_mesh(i, u)
            self.scene.add(m)
            if len(self.meshes) == 0:
                m.visible = True
            self.meshes.append(m)

    def goto(self, idx):
        if idx != self._last_idx:
            self.meshes[idx].visible = True
            if self._last_idx:
                self.meshes[self._last_idx].visible = False
            self.renderer.render(self.scene, self.cam)
            self._last_idx = idx

    def freeze_camera(self, freeze=True):
        self.controller.enablePan = not freeze
        self.controller.enableZoom = not freeze
        self.controller.enableRotate = not freeze

    def _setup_scene(self, bounding_box, render_size):
        self.min_width = render_size[0]
        self.min_height = render_size[1]
        fov_angle = 60
        if len(bounding_box[0]) == 2:
            lower = np.array([bounding_box[0][0], bounding_box[0][1], 0])
            upper = np.array([bounding_box[1][0], bounding_box[1][1], 0])
            bounding_box = (lower, upper)
        combined_bounds = np.hstack(bounding_box)

        absx = np.abs(combined_bounds[0] - combined_bounds[3])
        not_mathematical_distance_scaling = 1.2
        self.camera_distance = (np.sin((90 - fov_angle / 2) * np.pi / 180) * 0.5
                                * absx / np.sin(fov_angle / 2 * np.pi / 180))
        self.camera_distance *= not_mathematical_distance_scaling
        xhalf = (combined_bounds[0] + combined_bounds[3]) / 2
        yhalf = (combined_bounds[1] + combined_bounds[4]) / 2
        zhalf = (combined_bounds[2] + combined_bounds[5]) / 2
        self.mesh_center = (xhalf, yhalf, zhalf)
        self.cam = p3js.PerspectiveCamera(aspect=render_size[0] / render_size[1],
                                          position=[0, 0, 0 + self.camera_distance])
        self.light = p3js.AmbientLight(color='white', intensity=1.0)
        self.scene = p3js.Scene(children=[self.cam, self.light], background='white')
        self.controller = p3js.OrbitControls(controlling=self.cam, position=[0, 0, 0 + self.camera_distance],
                                             target=[0, 0, 0])
        self.freeze_camera(True)
        self.renderer = p3js.Renderer(camera=self.cam, scene=self.scene,
                                      controls=[self.controller], webgl_version=1,
                                      width=render_size[0], height=render_size[1])
        self.renderer.layout.max_height = f'{render_size[1]}px'


class ColorBarRenderer(widgets.HBox):
    def __init__(self, render_size, color_map, vmin=None, vmax=None):
        self.render_size = render_size
        self.layout = Layout(min_width=str(render_size[0]), min_height=str(render_size[1]), margin='10px ')
        self.color_map = color_map
        self.vmin, self.vmax = vmin, vmax
        self.image, labels = self._gen_sprite()
        super().__init__(children=[self.image, labels])

    def freeze_camera(self, freeze=True):
        pass

    def goto(self, idx):
        labels = self.labels.children
        text_fmt = '{:+1.3e}'
        labels[0].value = text_fmt.format(self.vmax[idx])
        labels[1].value = text_fmt.format((self.vmax[idx]+self.vmin[idx])/2)
        labels[2].value = text_fmt.format(self.vmin[idx])

    def _gen_sprite(self):
        from PIL import Image, ImageDraw
        # upsacle to pow2
        bar_width = 25
        sprite_size = (bar_width, self.render_size[1])
        image = Image.new('RGBA', sprite_size, color=(255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        bar_height = sprite_size[1]
        # we have to flip the Y coord cause PIL's coordinate system is different from OGL
        for i in range(bar_height):
            cl = tuple((np.array(self.color_map(bar_height-i))*255).astype(np.int_))
            draw.line([(0, i), (bar_width, i)], cl, width=1)

        label_layout = Layout(margin='0px 2px')
        self.labels = widgets.VBox([Label('', layout=label_layout),
                                    Label('', layout=label_layout),
                                    Label('', layout=label_layout)],
                                   layout=Layout(justify_content='space-between'))
        self.goto(0)

        of = BytesIO()
        image.save(of, format='png')
        of.seek(0)
        return [widgets.Image(
            value=of.read(),
            format='png',
            width=bar_width,
            height=self.render_size[1],
            layout=Layout(margin='0px'),
        ), self.labels]


class ThreeJSPlot(widgets.VBox):
    def __init__(self, grid, color_map, title, bounding_box, codim, U, vmins, vmaxs, separate_colorbars, size):
        render_size = (300, 300)
        self.renderer = [Renderer(u, grid, render_size, color_map, title, bounding_box=bounding_box, codim=codim,
                                  vmin=vmin, vmax=vmax)
                         for u, vmin, vmax in zip(U, vmins, vmaxs)]
        bar_size = (100, render_size[1])
        if not separate_colorbars:
            self.colorbars = [ColorBarRenderer(render_size=bar_size, vmin=vmins[0], vmax=vmaxs[0], color_map=color_map)]
            # prevent line break between last plot and colorbar
            self.r_hbox_items = self.renderer[:-1]
            self.r_hbox_items.append(widgets.HBox([self.renderer[-1], self.colorbars[0]]))
        else:
            self.r_hbox_items = []
            self.colorbars = []
            for vmin, vmax, renderer in zip(vmins, vmaxs, self.renderer):
                cr = ColorBarRenderer(render_size=bar_size, vmin=vmin, vmax=vmax, color_map=color_map)
                self.r_hbox_items.append(widgets.HBox([renderer, cr]))
                self.colorbars.append(cr)
        layout = Layout(display='flex', flex_flow='row wrap', align_items='stretch', justify_content='flex-start',)
        children = [widgets.Box(self.r_hbox_items, layout=layout)]
        if size > 1:
            def _goto_idx(idx):
                for c in self.renderer:
                    c.goto(idx)
                for c in self.colorbars:
                    c.goto(idx)
            play = Play(min=0, max=size - 1, step=1, value=0, description='Timestep:', layout=Layout(margin='0px'))
            interact(idx=play).widget(_goto_idx)
            slider = IntSlider(min=0, max=size - 1, step=1, value=0, description='Timestep:')
            widgets.jslink((play, 'value'), (slider, 'value'))
            controls = widgets.HBox([play, slider], layout=Layout(margin='0px 10px'))
            children.append(controls)

        super().__init__(children=children)

    async def finish_loading(self):
        while not all(r.load.done() for r in self.renderer):
            await asyncio.sleep(1)


def visualize_py3js(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, columns=2, color_map=get_cmap('viridis')):
    """Generate a pythreejs plot and associated controls for scalar data associated to a 2D |Grid|.

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
    assert isinstance(U, VectorArray) \
           or (isinstance(U, tuple) and all(isinstance(u, VectorArray) for u in U)
               and all(len(u) == len(U[0]) for u in U))
    if isinstance(U, VectorArray):
        size = len(U)
        U = (U.to_numpy().astype(np.float32, copy=False),)
    else:
        size = len(U[0])
        U = tuple(u.to_numpy().astype(np.float32, copy=False) for u in U)

    if rescale_colorbars:
        vmins = tuple(np.min(u, axis=1) for u in U)
        vmaxs = tuple(np.max(u, axis=1) for u in U)
    else:
        vmins = tuple(np.repeat(np.min(u), len(u)) for u in U)
        vmaxs = tuple(np.repeat(np.max(u), len(u)) for u in U)
    if not separate_colorbars:
        vmins = (np.min(np.vstack(vmins), axis=0),) * len(U)
        vmaxs = (np.max(np.vstack(vmaxs), axis=0),) * len(U)

    return ThreeJSPlot(grid, color_map, title, bounding_box, codim, U, vmins, vmaxs, separate_colorbars, size)
