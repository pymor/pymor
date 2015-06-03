# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.tools import mpi
from pymor.vectorarrays.interfaces import VectorArrayInterface


class MPIVectorArray(VectorArrayInterface):

    def __init__(self, cls, subtype, obj_id):
        self.cls = cls
        self.array_subtype = subtype
        self.obj_id = obj_id

    @classmethod
    def make_array(cls, subtype, count=0, reserve=0):
        return MPIVectorArray(subtype[0], subtype[1],
                              mpi.call(_make_array, subtype[0], subtype=subtype[1], count=count, reserve=reserve))

    def __len__(self):
        return mpi.call(mpi.method_call, self.obj_id, '__len__')

    @property
    def dim(self):
        return mpi.call(_dim, self.obj_id)

    @property
    def subtype(self):
        return (self.cls, self.array_subtype)

    def copy(self, ind=None):
        return MPIVectorArray(self.cls, self.array_subtype,
                              mpi.call(mpi.method_call_manage, self.obj_id, 'copy', ind=ind))

    def append(self, other, o_ind=None, remove_from_other=False):
        mpi.call(mpi.method_call1, self.obj_id, 'append', other.obj_id,
                 o_ind=o_ind, remove_from_other=remove_from_other)

    def remove(self, ind=None):
        mpi.call(mpi.method_call, self.obj_id, 'remove', ind=ind)

    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        mpi.call(mpi.method_call1, self.obj_id, 'replace', other.obj_id,
                 ind=ind, o_ind=o_ind, remove_from_other=remove_from_other)

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        return mpi.call(mpi.method_call1, self.obj_id, 'almost_equal', other.obj_id,
                        ind=ind, o_ind=o_ind, rtol=rtol, atol=atol)

    def scal(self, alpha, ind=None):
        mpi.call(mpi.method_call, self.obj_id, 'scal', alpha, ind=ind)

    def axpy(self, alpha, x, ind=None, x_ind=None):
        mpi.call(_axpy, self.obj_id, alpha, x.obj_id, ind=ind, x_ind=x_ind)

    def dot(self, other, ind=None, o_ind=None):
        return mpi.call(mpi.method_call1, self.obj_id, 'dot', other.obj_id, ind=ind, o_ind=o_ind)

    def pairwise_dot(self, other, ind=None, o_ind=None):
        return mpi.call(mpi.method_call1, self.obj_id, 'pairwise_dot', other.obj_id, ind=ind, o_ind=o_ind)

    def lincomb(self, coefficients, ind=None):
        return MPIVectorArray(self.cls, self.array_subtype,
                              mpi.call(mpi.method_call_manage, self.obj_id, 'lincomb', coefficients, ind=ind))

    def l1_norm(self, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'l1_norm', ind=ind)

    def l2_norm(self, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'l2_norm', ind=ind)

    def components(self, component_indices, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'components', component_indices, ind=ind)

    def amax(self, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'amax', ind=ind)

    def __del__(self):
        mpi.call(mpi.remove_object, self.obj_id)


class MPIDistributed(object):

    @property
    def dim(self):
        dims = mpi.comm.gather(super(MPIDistributed, self).dim, root=0)
        if mpi.rank0:
            return sum(dims)

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        local_results = super(MPIDistributed, self).almost_equal(other, ind=ind, o_ind=o_ind,
                                                                 rtol=rtol, atol=atol).astype(np.int8)
        results = np.empty((mpi.size, len(local_results)), dtype=np.int8) if mpi.rank0 else None
        mpi.comm.Gather(local_results, results, root=0)
        if mpi.rank0:
            return np.all(results, axis=0)

    def dot(self, other, ind=None, o_ind=None):
        local_results = super(MPIDistributed, self).dot(other, ind=ind, o_ind=o_ind)
        assert local_results.dtype == np.float64
        results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
        mpi.comm.Gather(local_results, results, root=0)
        if mpi.rank0:
            return np.sum(results, axis=0)

    def pairwise_dot(self, other, ind=None, o_ind=None):
        local_results = super(MPIDistributed, self).pairwise_dot(other, ind=ind, o_ind=o_ind)
        assert local_results.dtype == np.float64
        results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
        mpi.comm.Gather(local_results, results, root=0)
        if mpi.rank0:
            return np.sum(results, axis=0)

    def l1_norm(self, ind=None):
        local_results = super(MPIDistributed, self).l1_norm(ind=ind)
        assert local_results.dtype == np.float64
        results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
        mpi.comm.Gather(local_results, results, root=0)
        if mpi.rank0:
            return np.sum(results, axis=0)

    def l2_norm(self, ind=None):
        local_results = super(MPIDistributed, self).l2_norm(ind=ind)
        assert local_results.dtype == np.float64
        results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
        mpi.comm.Gather(local_results, results, root=0)
        if mpi.rank0:
            return np.sqrt(np.sum(results ** 2, axis=0))

    def components(self, component_indices, ind=None):
        raise NotImplementedError

    def amax(self, ind=None):
        raise NotImplementedError


class MPIForbidCommunication(object):

    @property
    def dim(self):
        raise NotImplementedError

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        raise NotImplementedError

    def dot(self, other, ind=None, o_ind=None):
        raise NotImplementedError

    def pairwise_dot(self, other, ind=None, o_ind=None):
        raise NotImplementedError

    def l1_norm(self, ind=None):
        raise NotImplementedError

    def l2_norm(self, ind=None):
        raise NotImplementedError

    def components(self, component_indices, ind=None):
        raise NotImplementedError

    def amax(self, ind=None):
        raise NotImplementedError


class MPILocalSubtypes(object):

    @property
    def subtype(self):
        subtypes = mpi.comm.gather(super(MPILocalSubtypes, self).subtype, root=0)
        if mpi.rank0:
            return tuple(subtypes)

    @classmethod
    def make_array(cls, subtype, count=0, reserve=0):
        return super(MPILocalSubtypes, cls).make_array(subtype[mpi.rank], count=count, reserve=reserve)


def _make_array(cls, subtype=None, count=0, reserve=0):
    obj = cls.make_array(subtype=subtype, count=count, reserve=reserve)
    return mpi.manage_object(obj)


def _dim(obj_id):
    obj = mpi.get_object(obj_id)
    return obj.dim


def _axpy(obj_id, alpha, x_obj_id, ind=None, x_ind=None):
    obj = mpi.get_object(obj_id)
    x = mpi.get_object(x_obj_id)
    obj.axpy(alpha, x, ind=ind, x_ind=x_ind)
