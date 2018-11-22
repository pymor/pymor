# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.tools import mpi
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.vectorarrays.mpi import MPIVectorSpaceAutoComm


def random_array(dims, length, seed):
    if isinstance(dims, Number):
        dims = (dims,)
    return MPIVectorSpaceAutoComm(tuple(NumpyVectorSpace(dim) for dim in dims)).make_array(
        mpi.call(_random_array, dims, length, seed)
    )


def _random_array(dims, length, seed):
    np.random.seed(seed + mpi.rank)
    dim = dims[mpi.rank] if len(dims) > 1 else dims[0]
    array = NumpyVectorSpace.make_array(np.random.random((length, dim)))
    obj_id = mpi.manage_object(array)
    return obj_id
