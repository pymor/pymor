# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module provides plotting support inside the Jupyter notebook.

To use these routines you first have to execute ::

        %matplotlib notebook

inside the given notebook.
"""
import IPython

from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.discretizers.builtin.gui.jupyter.matplotlib import visualize_patch
from pymor.core.config import config

# AFAICT there is no robust way to query for loaded extensions
# and we have to make sure we do not setup two redirects
_extension_loaded = False


@defaults('backend')
def get_visualizer(backend='pyvista'):
    if backend not in ('py3js', 'pyvista', 'MPL'):
        raise ValueError
    if backend == 'py3js' and config.HAVE_PYTHREEJS:
        from pymor.discretizers.builtin.gui.jupyter.threejs import visualize_py3js
        return visualize_py3js
    elif backend == 'pyvista' and config.HAVE_PYVISTA:
        from pymor.discretizers.builtin.gui.jupyter.vista import visualize_vista_vectorarray
        return visualize_vista_vectorarray
    if backend != 'MPL':
        msg = f'Selected {backend} as visualizer not available. Falling back to matplotlib'
        getLogger('pymor.discretizers.builtin.gui.jupyter').warning(msg=msg)
    return visualize_patch


def progress_bar(sequence, every=None, size=None, name='Parameters'):
    from ipywidgets import IntProgress, HTML, VBox
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
    IPython.display.display(box)

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


def load_ipython_extension(ipython):
    global _extension_loaded
    from pymor.discretizers.builtin.gui.jupyter.logging import redirect_logging
    ipython.events.register('pre_run_cell', redirect_logging.start)
    ipython.events.register('post_run_cell', redirect_logging.stop)
    _extension_loaded = True


def unload_ipython_extension(ipython):
    global _extension_loaded
    from pymor.discretizers.builtin.gui.jupyter.logging import redirect_logging
    ipython.events.unregister('pre_run_cell', redirect_logging.start)
    ipython.events.unregister('post_run_cell', redirect_logging.stop)
    _extension_loaded = False
