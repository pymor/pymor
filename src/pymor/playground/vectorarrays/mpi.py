# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.tools import mpi
from pymor.vectorarrays.numpy import NumpyVectorArray


class MPINumpyVectorArray(NumpyVectorArray):

    @property
    def dim(self):
        dims = mpi.comm.gather(self._array.shape[1], root=0)
        if mpi.rank0:
            return sum(dims)

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        local_results = super(MPINumpyVectorArray, self).almost_equal(other, ind=ind, o_ind=o_ind,
                                                                      rtol=rtol, atol=atol).astype(np.int8)
        results = np.empty((mpi.size, len(local_results)), dtype=np.int8) if mpi.rank0 else None
        mpi.comm.Gather(local_results, results, root=0)
        if mpi.rank0:
            return np.all(results, axis=0)

    def dot(self, other, ind=None, o_ind=None):
        local_results = super(MPINumpyVectorArray, self).dot(other, ind=ind, o_ind=o_ind)
        assert local_results.dtype == np.float64
        results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
        mpi.comm.Gather(local_results, results, root=0)
        if mpi.rank0:
            return np.sum(results, axis=0)

    def pairwise_dot(self, other, ind=None, o_ind=None):
        local_results = super(MPINumpyVectorArray, self).pairwise_dot(other, ind=ind, o_ind=o_ind)
        assert local_results.dtype == np.float64
        results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
        mpi.comm.Gather(local_results, results, root=0)
        if mpi.rank0:
            return np.sum(results, axis=0)

    def l1_norm(self, ind=None):
        local_results = super(MPINumpyVectorArray, self).l1_norm(ind=ind)
        assert local_results.dtype == np.float64
        results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
        mpi.comm.Gather(local_results, results, root=0)
        if mpi.rank0:
            return np.sum(results, axis=0)

    def l2_norm(self, ind=None):
        local_results = super(MPINumpyVectorArray, self).l2_norm(ind=ind)
        assert local_results.dtype == np.float64
        results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
        mpi.comm.Gather(local_results, results, root=0)
        if mpi.rank0:
            return np.sqrt(np.sum(results ** 2, axis=0))

    def components(self, component_indices, ind=None):
        raise NotImplementedError

    def amax(self, ind=None):
        raise NotImplementedError


# for debugging
def random_array(dim, length, seed):
    from pymor.vectorarrays.mpi import MPIVectorArray
    return MPIVectorArray(MPINumpyVectorArray, dim,
                          mpi.call(_random_array, dim, length, seed))


def _random_array(dim, length, seed):
    np.random.seed(seed + mpi.rank)
    array = MPINumpyVectorArray(np.random.random((length, dim)))
    obj_id = mpi.manage_object(array)
    return obj_id


def set_data(obj_id, rank, data):
    if rank == mpi.rank:
        obj = mpi.get_object(obj_id)
        obj.data[:] = data
