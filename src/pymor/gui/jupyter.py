# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module provides plotting support inside the Jupyter notebook.

To use these routines you first have to execute ::

        %matplotlib notebook

inside the given notebook.
"""

import numpy as np

from pymor.core import logger
from pymor.core.config import config
from pymor.core.logger import ColoredFormatter
from pymor.gui.matplotlib import MatplotlibPatchAxes
from pymor.vectorarrays.interfaces import VectorArrayInterface
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display
import contextlib
import ipywidgets


def visualize_patch(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, columns=2):
    """Visualize scalar data associated to a two-dimensional |Grid| as a patch plot.

    The grid's |ReferenceElement| must be the triangle or square. The data can either
    be attached to the faces or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    U
        |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
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
    """

    assert isinstance(U, VectorArrayInterface) \
        or (isinstance(U, tuple)
            and all(isinstance(u, VectorArrayInterface) for u in U)
            and all(len(u) == len(U[0]) for u in U))
    U = (U.to_numpy().astype(np.float64, copy=False),) if isinstance(U, VectorArrayInterface) else \
        tuple(u.to_numpy().astype(np.float64, copy=False) for u in U)

    if not config.HAVE_MATPLOTLIB:
        raise ImportError('cannot visualize: import of matplotlib failed')
    if not config.HAVE_IPYWIDGETS and len(U[0]) > 1:
        raise ImportError('cannot visualize: import of ipywidgets failed')

    if isinstance(legend, str):
        legend = (legend,)
    assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)
    if len(U) < 2:
        columns = 1

    class Plot:

        def __init__(self):
            if separate_colorbars:
                if rescale_colorbars:
                    self.vmins = tuple(np.min(u[0]) for u in U)
                    self.vmaxs = tuple(np.max(u[0]) for u in U)
                else:
                    self.vmins = tuple(np.min(u) for u in U)
                    self.vmaxs = tuple(np.max(u) for u in U)
            else:
                if rescale_colorbars:
                    self.vmins = (min(np.min(u[0]) for u in U),) * len(U)
                    self.vmaxs = (max(np.max(u[0]) for u in U),) * len(U)
                else:
                    self.vmins = (min(np.min(u) for u in U),) * len(U)
                    self.vmaxs = (max(np.max(u) for u in U),) * len(U)

            import matplotlib.pyplot as plt

            rows = int(np.ceil(len(U) / columns))
            self.figure = figure = plt.figure()

            self.plots = plots = []
            axes = []
            for i, (vmin, vmax) in enumerate(zip(self.vmins, self.vmaxs)):
                ax = figure.add_subplot(rows, columns, i+1)
                axes.append(ax)
                plots.append(MatplotlibPatchAxes(figure, grid, bounding_box=bounding_box, vmin=vmin, vmax=vmax,
                                                 codim=codim, colorbar=separate_colorbars))
                if legend:
                    ax.set_title(legend[i])

            plt.tight_layout()
            if not separate_colorbars:
                figure.colorbar(plots[0].p, ax=axes)

        def set(self, U, ind):
            if rescale_colorbars:
                if separate_colorbars:
                    self.vmins = tuple(np.min(u[ind]) for u in U)
                    self.vmaxs = tuple(np.max(u[ind]) for u in U)
                else:
                    self.vmins = (min(np.min(u[ind]) for u in U),) * len(U)
                    self.vmaxs = (max(np.max(u[ind]) for u in U),) * len(U)

            for u, plot, vmin, vmax in zip(U, self.plots, self.vmins, self.vmaxs):
                plot.set(u[ind], vmin=vmin, vmax=vmax)

    plot = Plot()
    plot.set(U, 0)

    if len(U[0]) > 1:

        from ipywidgets import interact, IntSlider

        def set_time(t):
            plot.set(U, t)

        interact(set_time, t=IntSlider(min=0, max=len(U[0])-1, step=1, value=0))

    return plot



def progress_bar(sequence, every=None, size=None, name='Parameters'):
    # c&p from https://github.com/kuk/log-progress
    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = f"{name}: {str(index or '?')}"


@contextlib.contextmanager
def redirect_logging():
    # TODO: this should be usable via cell magics too
    import logging

    class LogViewer(logging.Handler):
        out = None

        def __init__(self, out):
            super().__init__()
            self.out = out
            self.setFormatter(ColoredFormatter())

        def emit(self, record):
            record = self.format(record)
            self.out.outputs = ({'name': 'stdout', 'output_type': 'stream', 'text': f'{logger.BLACK}{record}\n'},) + self.out.outputs

        def __repr__(self):
            return '<%s %s>' % (self.__class__.__name__, self.out)

    # TODO: the widget needs to be fancier, maybe with fold stuff, and correct scroll direction
    out = ipywidgets.Output(layout=ipywidgets.Layout(width='100%', height='8em', border='solid'))
    new_handler = LogViewer(out)

    def _new_default(_):
        return [new_handler]

    old_default = logger.default_handler
    logger.default_handler = _new_default
    old_handlers = {name: logging.getLogger(name).handlers for name in logging.root.manager.loggerDict}
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = [new_handler]

    display(out)

    yield

    logger.default_handler = old_default
    for name in logging.root.manager.loggerDict:
        try:
            logging.getLogger(name).handlers = old_handlers[name]
        except KeyError:
            # loggers that have been created during the redirect get a default handler
            logging.getLogger(name).handlers = logger.default_handler()