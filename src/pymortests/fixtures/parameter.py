# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.parameters.base import Mu


def mu_of_type(parameters, seed):
    np.random.seed(seed)
    while True:
        if parameters is None:
            yield None
        else:
            yield Mu({k: np.random.random(v) for k, v in parameters.items()})
