# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.parameters.base import Parameter


def parameters_of_type(parameter_type, seed):
    np.random.seed(seed)
    while True:
        if parameter_type is None:
            yield None
        else:
            yield Parameter({k: np.random.random(v) for k, v in parameter_type.items()})
