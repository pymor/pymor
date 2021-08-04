# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.vectorarrays.interface import VectorArray


def vmin_vmax_numpy(U, separate_colorbars, rescale_colorbars):
    if separate_colorbars:
        if rescale_colorbars:
            vmins = tuple(np.min(u[0]) for u in U)
            vmaxs = tuple(np.max(u[0]) for u in U)
        else:
            vmins = tuple(np.min(u) for u in U)
            vmaxs = tuple(np.max(u) for u in U)
    else:
        if rescale_colorbars:
            vmins = (min(np.min(u[0]) for u in U),) * len(U)
            vmaxs = (max(np.max(u[0]) for u in U),) * len(U)
        else:
            vmins = (min(np.min(u) for u in U),) * len(U)
            vmaxs = (max(np.max(u) for u in U),) * len(U)
    return vmins, vmaxs


def vmin_vmax_vectorarray(U, separate_colorbars, rescale_colorbars):
    Unp = (U.to_numpy().astype(np.float64, copy=False),) if isinstance(U, VectorArray) else \
        tuple(u.to_numpy().astype(np.float64, copy=False) for u in U)
    return vmin_vmax_numpy(Unp, separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars)