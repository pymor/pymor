import multiprocessing
import numpy as np

from pymor.vectorarrays.interfaces import VectorArrayInterface

from pymor.gui.kivy_frontend.widgets.matplotlib_widget import getMatplotlibOnedWidget, getMatplotlibPatchWidget,\
    HAVE_KIVY, HAVE_MATPLOTLIB
from pymor.gui.kivy_frontend.widgets.gl_widget import getGLPatchWidget
from pymor.gui.kivy_frontend.windows import getPlotMainWindow, getPatchPlotMainWindow
from pymor.core.defaults import defaults


def visualize_oned(grid, U, codim=1, title=None, legend=None, separate_plots=False, block=False):

    assert isinstance(U, VectorArrayInterface) and hasattr(U, 'data') \
                or (isinstance(U, tuple) and all(isinstance(u, VectorArrayInterface) and hasattr(u, 'data') for u in U)
                    and all(len(u) == len(U[0]) for u in U))
    U = (U.data,) if hasattr(U, 'data') else tuple(u.data for u in U)
    if isinstance(legend, str):
        legend = (legend,)
    assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)

    _launch_kivy_app(lambda: getPlotMainWindow(U,
                                               getMatplotlibOnedWidget(None, grid, count=len(U), vmin=[np.min(u) for u in U],
                                                                       vmax=[np.max(u) for u in U], legend=legend, codim=codim,
                                                                       separate_plots=separate_plots),
                                               length=len(U[0]),
                                               title=title, isNotMatplotlib=False), block)


@defaults('backend', sid_ignore=('backend',))
def visualize_patch(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, backend='gl', block=False, columns=2):
    if not HAVE_KIVY:
        raise ImportError('cannot visualize: import of kivy failed')

    assert backend in {'gl', 'matplotlib'}

    if backend == 'matplotlib' and not HAVE_MATPLOTLIB:
        raise ImportError('cannot visualize: import of matplotlib failed')

    assert isinstance(U, VectorArrayInterface) and hasattr(U, 'data')\
        or (isinstance(U, tuple) and all(isinstance(u, VectorArrayInterface) and hasattr(u, 'data') for u in U)
        and all(len(u) == len(U[0]) for u in U))
    U = (U.data,) if hasattr(U, 'data') else tuple(u.data for u in U)
    if isinstance(legend, str):
        legend = (legend,)
    assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)

    if backend == 'gl':
        widget = getGLPatchWidget
    elif backend == 'matplotlib':
        widget = getMatplotlibPatchWidget
    else:
        raise ValueError("Only gl or matplotlib are supported")

    _launch_kivy_app(lambda: getPatchPlotMainWindow(U, grid=grid, bounding_box=bounding_box, codim=codim, title=title,
                                                    legend=legend, separate_colorbars=separate_colorbars,
                                                    rescale_colorbars=rescale_colorbars, columns=columns, widget=widget,
                                                    backend=backend), block=block)


_launch_kivy_app_pids = set()


def _launch_kivy_app(main_window_factory, block):

    def doit():
        main_window_factory().run()

    # Only one kivy app can be running in one process at a time.
    # Starting another kivy app after shutting one down doesn't seem to be supported either,
    # so run the kivy app in a new process no matter what.
    p = multiprocessing.Process(target=doit)
    p.start()
    _launch_kivy_app_pids.add(p.pid)
    if block:
        while p.is_alive():
            continue