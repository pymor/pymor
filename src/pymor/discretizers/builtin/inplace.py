# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
from pymor.core.exceptions import CythonExtensionNotBuiltError


def iadd_masked(U, V, U_ind):
    """This cython function is defined in pymor/discretizers/builtin/inplace.pyx."""
    raise CythonExtensionNotBuiltError()


def isub_masked(U, V, U_ind):
    """This cython function is defined in pymor/discretizers/builtin/inplace.pyx."""
    raise CythonExtensionNotBuiltError()
