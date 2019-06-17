import time
import warnings
import IPython
from ipywidgets import IntSlider, interact, widgets, Play
from k3d.objects import Mesh
from k3d.plot import Plot as k3dPlot
import k3d
import numpy as np
from k3d.transform import process_transform_arguments
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap

from pymor.grids.constructions import flatten_grid


class VectorArrayPlot(k3dPlot):
    def __init__(self, U, grid, codim, color_map, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'transform' in kwargs.keys():
            raise RuntimeError('supplying transforms is currently not supported for time series Data')

        self.subentities, self.coordinates, entity_map = flatten_grid(grid)
        self.data = (U.to_numpy() if codim == 0 else U.to_numpy()[:, entity_map].copy()).astype(np.float32)

        if grid.dim == 2:
            # pad 0 in z dimension
            self.vertices = np.zeros((len(self.coordinates), 3))
            self.vertices[:,:-1] = self.coordinates

        self.idx = 0
        self.mesh = process_transform_arguments(Mesh(vertices=np.array(self.vertices, np.float32),
             indices=np.array(self.subentities, np.uint32),
             color=k3d.k3d._default_color,
             opacity=1.0,
             attribute=self.data[self.idx],
             color_range=(np.nanmin(self.data), np.nanmax(self.data)),
             color_map=np.array(color_map, np.float32),
             wireframe=False,
             compression_level=0))
        self += self.mesh
        self.camera_no_pan = True
        self.camera_no_rotate = True
        self.camera_no_zoom = True

    def _goto_idx(self, idx):
        if idx > len(self.data) or idx < 0:
            warnings.warn(f'Index {idx} outside data range for VectorArrayPlot', RuntimeWarning)
            return
        self.idx = idx
        self.mesh.attribute = self.data[self.idx]

    def dec(self):
        self._goto_idx(self.idx-1)

    def inc(self):
        self._goto_idx(self.idx+1)


def plot(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, columns=2,
         color_map=get_cmap('viridis')):
    """Generate a k3d Plot and associated controls for  scalar data associated to a two-dimensional |Grid|.

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
        a Matplotlib Colormap object or a K3D array((step, r, g, b))
    """
    assert len(bounding_box) == 2
    assert all(1 < len(bounding_box[i]) < 4 for i in range(2))
    assert len(bounding_box[0]) == len(bounding_box[1])
    if isinstance(color_map, Colormap):
        color_map = [(x, *color_map(x)[:3]) for x in np.linspace(0, 1, 256)]

    if len(bounding_box[0]) == 2:
        lower = np.array([bounding_box[0][0], bounding_box[0][1], 0])
        upper = np.array([bounding_box[1][0], bounding_box[1][1], 0])
        bounding_box = (lower, upper)
    combined_bounds = np.hstack(bounding_box)

    vtkplot = VectorArrayPlot(U=U, grid=grid, codim=codim, color_attribute_name='None', grid_auto_fit=False,
                              camera_auto_fit=False, color_map=color_map)
    # display needs to have been called before changing camera/grid_visible
    vtkplot.display()
    # could be replaced with testing if the widget is'ready'
    time.sleep(0.5)
    vtkplot.grid_visible = False
    vtkplot.menu_visibility = False
    # guesstimate
    fov_angle = 30
    absx = np.abs(combined_bounds[0] - combined_bounds[3])
    c_dist = np.sin((90 - fov_angle) * np.pi / 180) * absx / (2 * np.sin(fov_angle * np.pi / 180))
    xhalf = (combined_bounds[0] + combined_bounds[3]) / 2
    yhalf = (combined_bounds[1] + combined_bounds[4]) / 2
    zhalf = (combined_bounds[2] + combined_bounds[5]) / 2
    # camera[posx, posy, posz, targetx, targety, targetz, upx, upy, upz]
    vtkplot.camera = (xhalf, yhalf, zhalf + c_dist,
                      xhalf, yhalf, zhalf,
                      0, 1, 0)
    size = len(U)
    if size > 1:
        play = Play(min=0, max=size - 1, step=1, value=0, description='Timestep:')
        interact(idx=play).widget(vtkplot._goto_idx)
        slider = IntSlider(min=0, max=size-1, step=1, value=0, description='Timestep:')
        interact(idx=slider).widget(vtkplot._goto_idx)
        widgets.jslink((play, 'value'), (slider, 'value'))
        hbox = widgets.HBox([play, slider])
        IPython.display.display(hbox)

    return vtkplot
