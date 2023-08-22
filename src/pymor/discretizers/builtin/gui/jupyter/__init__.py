# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module provides plotting support inside the Jupyter notebook."""

from packaging.version import parse as parse_version

from pymor.core.config import config
from pymor.core.defaults import defaults

# AFAICT there is no robust way to query for loaded extensions
# and we have to make sure we do not setup two redirects
_extension_loaded = False


@defaults('backend')
def get_visualizer(backend='prefer_k3d'):
    if backend not in ('MPL', 'k3d', 'prefer_k3d'):
        raise ValueError
    if backend == 'prefer_k3d':
        if config.HAVE_K3D:
            import k3d
            if parse_version(k3d.__version__) >= parse_version('2.15.4'):
                backend = 'k3d'
            else:
                backend = 'MPL'
        else:
            backend = 'MPL'
    if backend == 'k3d':
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
