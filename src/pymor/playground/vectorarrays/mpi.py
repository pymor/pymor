# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np

from pymor.tools import mpi
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.vectorarrays.mpi import MPIVectorArray, MPIDistributed


class MPINumpyVectorArray(MPIDistributed, NumpyVectorArray):
    pass


# for debugging
def random_array(dims, length, seed):
    if isinstance(dims, Number):
        dims = (dims,)
    return MPIVectorArray(MPINumpyVectorArray, tuple(dims),
                          mpi.call(_random_array, dims, length, seed))


def _random_array(dims, length, seed):
    np.random.seed(seed + mpi.rank)
    dim = dims[mpi.rank] if len(dims) > 1 else dims[0]
    array = MPINumpyVectorArray(np.random.random((length, dim)))
    obj_id = mpi.manage_object(array)
    return obj_id
