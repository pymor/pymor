# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.tools import mpi
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.vectorarrays.mpi import MPIVectorArrayAutoComm, MPIVectorAutoComm
from pymor.vectorarrays.list import ListVectorArray, NumpyVector


def random_array(dims, length, seed):
    if isinstance(dims, Number):
        dims = (dims,)
    return MPIVectorArrayAutoComm(NumpyVectorArray, tuple(dims),
                                  mpi.call(_random_array, dims, length, seed))


def _random_array(dims, length, seed):
    np.random.seed(seed + mpi.rank)
    dim = dims[mpi.rank] if len(dims) > 1 else dims[0]
    array = NumpyVectorArray(np.random.random((length, dim)))
    obj_id = mpi.manage_object(array)
    return obj_id


def random_list_array(dims, length, seed):
    if isinstance(dims, Number):
        dims = (dims,)
    return ListVectorArray([MPIVectorAutoComm(NumpyVector, tuple(dims),
                                              mpi.call(_random_vector, dims, seed + i))
                            for i in range(length)],
                           copy=False,
                           subtype=(MPIVectorAutoComm, (NumpyVector, tuple(dims))))


def _random_vector(dims, seed):
    np.random.seed(seed + mpi.rank)
    dim = dims[mpi.rank] if len(dims) > 1 else dims[0]
    vector = NumpyVector(np.random.random(dim), copy=False)
    obj_id = mpi.manage_object(vector)
    return obj_id
