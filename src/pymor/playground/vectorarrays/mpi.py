# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.tools import mpi
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.vectorarrays.mpi import MPIVectorArray, MPIDistributed, MPILocalSubtypes


class MPINumpyVectorArray(MPILocalSubtypes, MPIDistributed, NumpyVectorArray):
    pass


# for debugging
def random_array(dims, length, seed):
    return MPIVectorArray(MPINumpyVectorArray, tuple(dims),
                          mpi.call(_random_array, dims, length, seed))


def _random_array(dims, length, seed):
    np.random.seed(seed + mpi.rank)
    array = MPINumpyVectorArray(np.random.random((length, dims[mpi.rank])))
    obj_id = mpi.manage_object(array)
    return obj_id
