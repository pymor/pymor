# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

def iadd_masked(U, V, U_ind):
    """This cython function is defined in pymor/discretizers/builtin/inplace.pyx."""


def isub_masked(U, V, U_ind):
    """This cython function is defined in pymor/discretizers/builtin/inplace.pyx."""


raise ImportError('''
Cython extension module 'pymor.tools.inplace' has not been built.
Please run 'python setup.py build_ext --inplace' in the root
directory of the pyMOR repository.''')
