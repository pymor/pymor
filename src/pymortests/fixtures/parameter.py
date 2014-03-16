# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.parameters import Parameter


def parameter_of_type(parameter_type, seed):
    np.random.seed(seed)
    if parameter_type is None:
        return None
    else:
        return Parameter({k: np.random.random(v) for k, v in parameter_type.iteritems()})
