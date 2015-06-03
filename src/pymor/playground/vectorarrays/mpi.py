# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np

from pymor.tools import mpi
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.vectorarrays.mpi import MPIVectorArray, MPIDistributed, MPIVector, MPIDistributedVector
from pymor.vectorarrays.list import ListVectorArray, NumpyVector


class MPINumpyVectorArray(MPIDistributed, NumpyVectorArray):
    pass


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


class MPINumpyVector(MPIDistributedVector, NumpyVector):
    pass


def random_list_array(dims, length, seed):
    if isinstance(dims, Number):
        dims = (dims,)
    return ListVectorArray([MPIVector(MPINumpyVector, tuple(dims),
                                      mpi.call(_random_vector, dims, seed + i))
                            for i in range(length)],
                           copy=False,
                           subtype=(MPIVector, (MPINumpyVector, tuple(dims))))


def _random_vector(dims, seed):
    np.random.seed(seed + mpi.rank)
    dim = dims[mpi.rank] if len(dims) > 1 else dims[0]
    vector = MPINumpyVector(np.random.random(dim), copy=False)
    obj_id = mpi.manage_object(vector)
    return obj_id
