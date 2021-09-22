# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np


def vmin_vmax_vectorarray(array_tuple, separate_colorbars, rescale_colorbars):
    """
    Parameters
    ----------
    separate_colorbars
        iff True, min/max are taken per element of the U tuple

    rescale_colorbars
        iff False, min/max are the same for all indices for all elements of the U tuple
    """
    assert isinstance(array_tuple, tuple)
    ind_count = len(array_tuple[0])
    tuple_size = len(array_tuple)
    limits = [None] * ind_count
    mins, maxs = [None] * ind_count, [None] * ind_count
    for ind in range(ind_count):
        mins[ind] = tuple(np.min(U[ind].to_numpy()) for U in array_tuple)
        maxs[ind] = tuple(np.max(U[ind].to_numpy()) for U in array_tuple)

    for ind in range(ind_count):
        if rescale_colorbars:
            if separate_colorbars:
                limits[ind] = mins[ind], maxs[ind]
            else:
                limits[ind] = (min(mins),) * tuple_size, (max(maxs),) * tuple_size
        else:
            if separate_colorbars:
                limits[ind] = mins[0], maxs[0]
            else:
                limits[ind] = ((min(np.min(U[0].to_numpy()) for U in array_tuple),) * tuple_size,
                               (max(np.max(U[0].to_numpy()) for U in array_tuple),) * tuple_size)
    return limits