# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from importlib import import_module


def import_class(qualname):
    *module, klass = qualname.split('.')
    module = '.'.join(module)
    module = import_module(module)
    return getattr(module, klass)
