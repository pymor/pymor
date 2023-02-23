# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module provides plotting support inside the Jupyter notebook.

To use these routines you first have to execute ::

        %matplotlib notebook

inside the given notebook.
"""
from pymor.core.config import config
from pymor.core.defaults import defaults

# AFAICT there is no robust way to query for loaded extensions
# and we have to make sure we do not setup two redirects
_extension_loaded = False


@defaults('backend')
def get_visualizer(backend='py3js'):
    if backend not in ('py3js', 'MPL', 'k3d'):
        raise ValueError
    if backend == 'py3js' and config.HAVE_PYTHREEJS:
        from pymor.discretizers.builtin.gui.jupyter.threejs import visualize_py3js
        return visualize_py3js
    elif backend == 'k3d':
        assert config.HAVE_K3D
        from pymor.discretizers.builtin.gui.jupyter.kthreed import visualize_k3d
        return visualize_k3d
    else:
        from pymor.discretizers.builtin.gui.jupyter.matplotlib import visualize_patch
        return visualize_patch


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
