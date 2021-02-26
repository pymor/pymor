# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.exceptions import CythonExtensionNotBuiltError


def compute_edges(faces, num_vertices):
    """This cython function is defined in pymor/discretizers/builtin/grids/_unstructured.pyx."""
    raise CythonExtensionNotBuiltError()
