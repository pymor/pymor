# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
import asyncio
from time import sleep
from pprint import pprint

import numpy as np

import IPython
import numpy as np
from ipywidgets import IntSlider, interact, widgets, Play
import pythreejs as p3js
from matplotlib.cm import get_cmap

from pymor.grids.referenceelements import triangle, square
from pymor.grids.constructions import flatten_grid
from pymor.vectorarrays.interfaces import VectorArrayInterface

# we should try to limit ourselves to webgl 1.0 here since 2.0 (draft) is not as widely supported
# https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API#Browser_compatibility
# version directives and such are preprended by threejs
EL_VS = """
    // Attribute variable that contains coordinates of the vertices.
    attribute float vertex_index;
    attribute float data;
    //uniform sampler2D data;
    varying float texcoord;

    void main()
    {
        texcoord = data;
        gl_Position =  projectionMatrix * 
                        modelViewMatrix * 
                        vec4(position,1.0);
    }
    """

EL_FS = """
    uniform sampler2D colormap;
    varying float texcoord;

    void main()
    {
        gl_FragColor = texture2D(colormap, vec2(texcoord, 0.));
    }
    """


def _normalize(u):
    # rescale to be in [0,1], scale nan to be the smallest value
    u -= np.nanmin(u)
    u /= np.nanmax(u)
    return np.nan_to_num(u)


class Renderer(widgets.VBox):
    def __init__(self, U, grid, render_size, color_map, title, vmin=None, vmax=None,
                 bounding_box=([0, 0], [1, 1]), codim=2):
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)

        subentities, coordinates, entity_map = flatten_grid(grid)
        data = (U if codim == 0 else U[:, entity_map]).astype(np.float32)

        if codim == 2:
            if grid.dim == 2:
                # zero-pad in Z direction
                vertices = np.zeros((len(coordinates), 3))
                vertices[:, :-1] = coordinates
            elif grid.dim == 3:
                vertices = coordinates
            else:
                raise NotImplementedError
            indices = subentities
        else:
            raise NotImplementedError

        max_tex_size = 512
        cm = color_map(np.linspace(0,1, max_tex_size)).astype(np.float32)
        cm.resize((max_tex_size, 1, 4))
        color_map = p3js.DataTexture(cm, format='RGBAFormat',  width=max_tex_size, height=1, type='FloatType')
        uniforms=dict(
            colormap={'value': color_map, 'type': 'sampler2D'},
        )
        self.material = p3js.ShaderMaterial(vertexShader=EL_VS, fragmentShader=EL_FS, uniforms=uniforms,
                                            morphTargets=True, )

        self.buffer_vertices = p3js.BufferAttribute(vertices.astype(np.float32), normalized=False)
        self.buffer_faces    = p3js.BufferAttribute(indices.astype(np.uint32).ravel(), normalized=False)
        self._setup_scene(bounding_box, render_size)
        self.load = asyncio.ensure_future(self._load_data(data))
        self._last_idx = None
        self.meshes = []
        super().__init__(children=[self.renderer, ])
        print('done init')

    def _get_mesh(self, u):
        future = asyncio.Future()
        data = p3js.BufferAttribute(_normalize(u), normalized=True)
        geo = p3js.BufferGeometry(
            index=self.buffer_faces,
            attributes=dict(
                position=self.buffer_vertices,
                data=data
            )
        )
        mesh = p3js.Mesh(geometry=geo, material=self.material)
        mesh.visible = False
        future.set_result(mesh)
        return future

    async def _load_data(self, data):
        for u in data:
            m = await self._get_mesh(u)
            self.scene.add(m)
            if len(self.meshes) == 0:
                m.visible = True
            self.meshes.append(m)
        print('done loading')

    def goto(self, idx):
        if idx != self._last_idx:
            self.meshes[idx].visible = True
            if self._last_idx:
                self.meshes[self._last_idx].visible = False
            self.renderer.render(self.scene, self.cam)
            self._last_idx = idx

    def _setup_scene(self, bounding_box, render_size):
        fov_angle = 60
        if len(bounding_box[0]) == 2:
            lower = np.array([bounding_box[0][0], bounding_box[0][1], 0])
            upper = np.array([bounding_box[1][0], bounding_box[1][1], 0])
            bounding_box = (lower, upper)
        combined_bounds = np.hstack(bounding_box)

        absx = np.abs(combined_bounds[0] - combined_bounds[3])
        not_mathematical_distance_scaling = 1.2
        c_dist = np.sin((90 - fov_angle / 2) * np.pi / 180) * 0.5 * absx / np.sin(fov_angle / 2 * np.pi / 180)
        c_dist *= not_mathematical_distance_scaling
        xhalf = (combined_bounds[0] + combined_bounds[3]) / 2
        yhalf = (combined_bounds[1] + combined_bounds[4]) / 2
        zhalf = (combined_bounds[2] + combined_bounds[5]) / 2

        self.cam = p3js.PerspectiveCamera(aspect=render_size[0] / render_size[1],
                                          position=[xhalf, yhalf, zhalf + c_dist],
                                          fov_angle=fov_angle)
        self.light = p3js.AmbientLight(color='white', intensity=1.0)
        self.scene = p3js.Scene(children=([self.cam, self.light]), background='white')
        self.controller = p3js.OrbitControls(controlling=self.cam, position=[xhalf, yhalf, zhalf + c_dist])
        self.controller.target = [xhalf, yhalf, zhalf]
        self.controller.exec_three_obj_method('update')
        self.renderer = p3js.Renderer(camera=self.cam, scene=self.scene,
                                      controls=[self.controller],
                                      width=render_size[0], height=render_size[1])


def visualize_py3js(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, columns=2,
         color_map=get_cmap('viridis')):
    """Generate a pythreejs Plot and associated controls for  scalar data associated to a two-dimensional |Grid|.

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

    render_size=(400,400)
    renderer = [Renderer(u, grid, render_size, color_map, title, vmin=vmin, vmax=vmax, bounding_box=bounding_box, codim=codim)
                for u, vmin, vmax in zip(U, vmins, vmaxs)]
    r_hbox = widgets.HBox(renderer)
    if size > 1:
        def _goto_idx(idx):
            for c in renderer:
                c.goto(idx)
        play = Play(min=0, max=size - 1, step=1, value=0, description='Timestep:')
        interact(idx=play).widget(_goto_idx)
        slider = IntSlider(min=0, max=size - 1, step=1, value=0, description='Timestep:')
        widgets.jslink((play, 'value'), (slider, 'value'))
        controls = widgets.HBox([play, slider])
        r_hbox = widgets.VBox([r_hbox, controls])
    IPython.display.display(r_hbox)
    return r_hbox
