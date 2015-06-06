# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.tools import mpi
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.list import VectorInterface


class MPIVectorArray(VectorArrayInterface):

    def __init__(self, cls, subtype, obj_id):
        self.cls = cls
        self.array_subtype = subtype
        self.obj_id = obj_id

    @classmethod
    def make_array(cls, subtype, count=0, reserve=0):
        return cls(subtype[0], subtype[1],
                   mpi.call(_MPIVectorArray_make_array,
                            subtype[0], subtype=subtype[1], count=count, reserve=reserve))

    def __len__(self):
        return mpi.call(mpi.method_call, self.obj_id, '__len__')

    @property
    def dim(self):
        return mpi.call(_MPIVectorArray_dim, self.obj_id)

    @property
    def subtype(self):
        return (self.cls, self.array_subtype)

    def copy(self, ind=None):
        return type(self)(self.cls, self.array_subtype,
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
        mpi.call(_MPIVectorArray_axpy, self.obj_id, alpha, x.obj_id, ind=ind, x_ind=x_ind)

    def dot(self, other, ind=None, o_ind=None):
        return mpi.call(mpi.method_call1, self.obj_id, 'dot', other.obj_id, ind=ind, o_ind=o_ind)

    def pairwise_dot(self, other, ind=None, o_ind=None):
        return mpi.call(mpi.method_call1, self.obj_id, 'pairwise_dot', other.obj_id, ind=ind, o_ind=o_ind)

    def lincomb(self, coefficients, ind=None):
        return type(self)(self.cls, self.array_subtype,
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


def _MPIVectorArray_make_array(cls, subtype=(None,), count=0, reserve=0):
    subtype = subtype[mpi.rank] if len(subtype) > 1 else subtype[0]
    obj = cls.make_array(subtype=subtype, count=count, reserve=reserve)
    return mpi.manage_object(obj)


def _MPIVectorArray_dim(obj_id):
    obj = mpi.get_object(obj_id)
    return obj.dim


def _MPIVectorArray_axpy(obj_id, alpha, x_obj_id, ind=None, x_ind=None):
    obj = mpi.get_object(obj_id)
    x = mpi.get_object(x_obj_id)
    obj.axpy(alpha, x, ind=ind, x_ind=x_ind)


class MPIVectorArrayNoComm(MPIVectorArray):

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


class MPIVectorArrayAutoComm(MPIVectorArray):

    @property
    def dim(self):
        dim = getattr(self, '_dim', None)
        if dim is None:
            dim = self._get_dims()[0]
        return dim

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        return mpi.call(_MPIVectorArrayAutoComm_almost_equal, self.obj_id, other.obj_id,
                        ind=ind, o_ind=o_ind, rtol=rtol, atol=atol)

    def dot(self, other, ind=None, o_ind=None):
        return mpi.call(_MPIVectorArrayAutoComm_dot, self.obj_id, other.obj_id, ind=ind, o_ind=o_ind)

    def pairwise_dot(self, other, ind=None, o_ind=None):
        return mpi.call(_MPIVectorArrayAutoComm_pairwise_dot, self.obj_id, other.obj_id, ind=ind, o_ind=o_ind)

    def l1_norm(self, ind=None):
        return mpi.call(_MPIVectorArrayAutoComm_l1_norm, self.obj_id, ind=ind)

    def l2_norm(self, ind=None):
        return mpi.call(_MPIVectorArrayAutoComm_l2_norm, self.obj_id, ind=ind)

    def components(self, component_indices, ind=None):
        offsets = getattr(self, '_offsets', None)
        if offsets is None:
            offsets = self._get_dims()[1]
        component_indices = np.array(component_indices)
        return mpi.call(_MPIVectorArrayAutoComm_components, self.obj_id, offsets, component_indices, ind=ind)

    def amax(self, ind=None):
        offsets = getattr(self, '_offsets', None)
        if offsets is None:
            offsets = self._get_dims()[1]
        inds, vals = mpi.call(_MPIVectorArrayAutoComm_amax, self.obj_id, ind=ind)
        inds += offsets[:, np.newaxis]
        max_inds = np.argmax(vals, axis=0)
        return np.choose(max_inds, inds), np.choose(max_inds, vals)

    def _get_dims(self):
        dims = mpi.call(_MPIVectorArrayAutoComm_dim, self.obj_id)
        self._offsets = offsets = np.cumsum(np.concatenate(([0], dims)))[:-1]
        self._dim = dim = sum(dims)
        return dim, offsets


def _MPIVectorArrayAutoComm_dim(self):
    self = mpi.get_object(self)
    dims = mpi.comm.gather(self.dim, root=0)
    if mpi.rank0:
        return dims


def _MPIVectorArrayAutoComm_almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
    self = mpi.get_object(self)
    other = mpi.get_object(other)
    local_results = self.almost_equal(other, ind=ind, o_ind=o_ind, rtol=rtol, atol=atol).astype(np.int8)
    results = np.empty((mpi.size, len(local_results)), dtype=np.int8) if mpi.rank0 else None
    mpi.comm.Gather(local_results, results, root=0)
    if mpi.rank0:
        return np.all(results, axis=0)


def _MPIVectorArrayAutoComm_dot(self, other, ind=None, o_ind=None):
    self = mpi.get_object(self)
    other = mpi.get_object(other)
    local_results = self.dot(other, ind=ind, o_ind=o_ind)
    assert local_results.dtype == np.float64
    results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_results, results, root=0)
    if mpi.rank0:
        return np.sum(results, axis=0)


def _MPIVectorArrayAutoComm_pairwise_dot(self, other, ind=None, o_ind=None):
    self = mpi.get_object(self)
    other = mpi.get_object(other)
    local_results = self.pairwise_dot(other, ind=ind, o_ind=o_ind)
    assert local_results.dtype == np.float64
    results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_results, results, root=0)
    if mpi.rank0:
        return np.sum(results, axis=0)


def _MPIVectorArrayAutoComm_l1_norm(self, ind=None):
    self = mpi.get_object(self)
    local_results = self.l1_norm(ind=ind)
    assert local_results.dtype == np.float64
    results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_results, results, root=0)
    if mpi.rank0:
        return np.sum(results, axis=0)


def _MPIVectorArrayAutoComm_l2_norm(self, ind=None):
    self = mpi.get_object(self)
    local_results = self.l2_norm(ind=ind)
    assert local_results.dtype == np.float64
    results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_results, results, root=0)
    if mpi.rank0:
        return np.sqrt(np.sum(results ** 2, axis=0))


def _MPIVectorArrayAutoComm_components(self, offsets, component_indices, ind=None):
    self = mpi.get_object(self)
    offset = offsets[mpi.rank]
    dim = self.dim
    my_indices = np.logical_and(component_indices >= offset, component_indices < offset + dim)
    local_results = np.zeros((self.len_ind(ind), len(component_indices)))
    local_results[:, my_indices] = self.components(component_indices[my_indices] - offset, ind=ind)
    assert local_results.dtype == np.float64
    results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_results, results, root=0)
    if mpi.rank0:
        return np.sum(results, axis=0)


def _MPIVectorArrayAutoComm_amax(self, ind=None):
    self = mpi.get_object(self)
    local_inds, local_vals = self.amax(ind=ind)
    assert local_inds.dtype == np.int64
    assert local_vals.dtype == np.float64
    inds = np.empty((mpi.size,) + local_inds.shape, dtype=np.int64) if mpi.rank0 else None
    vals = np.empty((mpi.size,) + local_inds.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_inds, inds, root=0)
    mpi.comm.Gather(local_vals, vals, root=0)
    if mpi.rank0:
        return inds, vals


class MPIVector(VectorInterface):

    def __init__(self, cls, subtype, obj_id):
        self.cls = cls
        self.vector_subtype = subtype
        self.obj_id = obj_id

    @classmethod
    def make_zeros(cls, subtype=None):
        return cls(subtype[0], subtype[1],
                   mpi.call(_MPIVector_make_zeros, subtype[0], subtype=subtype[1]))

    @property
    def dim(self):
        return mpi.call(_MPIVectorArray_dim, self.obj_id)

    @property
    def subtype(self):
        return (self.cls, self.vector_subtype)

    def copy(self):
        return type(self)(self.cls, self.vector_subtype,
                          mpi.call(mpi.method_call_manage, self.obj_id, 'copy'))

    def almost_equal(self, other, rtol=None, atol=None):
        return mpi.call(mpi.method_call1, self.obj_id, 'almost_equal', other.obj_id,
                        rtol=rtol, atol=atol)

    def scal(self, alpha):
        mpi.call(mpi.method_call, self.obj_id, 'scal', alpha)

    def axpy(self, alpha, x):
        mpi.call(_MPIVector_axpy, self.obj_id, alpha, x.obj_id)

    def dot(self, other):
        return mpi.call(mpi.method_call1, self.obj_id, 'dot', other.obj_id)

    def l1_norm(self):
        return mpi.call(mpi.method_call, self.obj_id, 'l1_norm')

    def l2_norm(self):
        return mpi.call(mpi.method_call, self.obj_id, 'l2_norm')

    def components(self, component_indices):
        return mpi.call(mpi.method_call, self.obj_id, 'components', component_indices)

    def amax(self):
        return mpi.call(mpi.method_call, self.obj_id, 'amax')

    def __del__(self):
        mpi.call(mpi.remove_object, self.obj_id)


def _MPIVector_axpy(obj_id, alpha, x_obj_id):
    obj = mpi.get_object(obj_id)
    x = mpi.get_object(x_obj_id)
    obj.axpy(alpha, x)


def _MPIVector_make_zeros(cls, subtype=(None,)):
    subtype = subtype[mpi.rank] if len(subtype) > 1 else subtype[0]
    obj = cls.make_zeros(subtype)
    return mpi.manage_object(obj)


class MPIVectorNoComm(object):

    @property
    def dim(self):
        raise NotImplementedError

    def almost_equal(self, other, rtol=None, atol=None):
        raise NotImplementedError

    def dot(self, other):
        raise NotImplementedError

    def pairwise_dot(self, other):
        raise NotImplementedError

    def l1_norm(self):
        raise NotImplementedError

    def l2_norm(self):
        raise NotImplementedError

    def components(self, component_indices):
        raise NotImplementedError

    def amax(self):
        raise NotImplementedError


class MPIVectorAutoComm(MPIVector):

    @property
    def dim(self):
        return mpi.call(_MPIVectorAutoComm_dim, self.obj_id)

    def almost_equal(self, other, rtol=None, atol=None):
        return mpi.call(_MPIVectorAutoComm_almost_equal, self.obj_id, other.obj_id, rtol=rtol, atol=atol)

    def dot(self, other):
        return mpi.call(_MPIVectorAutoComm_dot, self.obj_id, other.obj_id)

    def l1_norm(self):
        return mpi.call(_MPIVectorAutoComm_l1_norm, self.obj_id)

    def l2_norm(self):
        return mpi.call(_MPIVectorAutoComm_l2_norm, self.obj_id)

    def components(self, component_indices):
        raise NotImplementedError

    def amax(self):
        raise NotImplementedError


def _MPIVectorAutoComm_dim(self):
    self = mpi.get_object(self)
    dims = mpi.comm.gather(self.dim, root=0)
    if mpi.rank0:
        return sum(dims)


def _MPIVectorAutoComm_almost_equal(self, other, rtol=None, atol=None):
    self = mpi.get_object(self)
    other = mpi.get_object(other)
    local_result = self.almost_equal(other, rtol=rtol, atol=atol).astype(np.int8)
    results = np.empty((mpi.size,), dtype=np.int8) if mpi.rank0 else None
    mpi.comm.Gather(local_result, results, root=0)
    if mpi.rank0:
        return np.all(results)


def _MPIVectorAutoComm_dot(self, other):
    self = mpi.get_object(self)
    other = mpi.get_object(other)
    local_result = self.dot(other)
    assert local_result.dtype == np.float64
    results = np.empty((mpi.size,), dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_result, results, root=0)
    if mpi.rank0:
        return np.sum(results)


def _MPIVectorAutoComm_l1_norm(self):
    self = mpi.get_object(self)
    local_result = self.l1_norm()
    assert local_result.dtype == np.float64
    results = np.empty((mpi.size,), dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_result, results, root=0)
    if mpi.rank0:
        return np.sum(results)


def _MPIVectorAutoComm_l2_norm(self):
    self = mpi.get_object(self)
    local_result = self.l2_norm()
    assert local_result.dtype == np.float64
    results = np.empty((mpi.size,), dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_result, results, root=0)
    if mpi.rank0:
        return np.sqrt(np.sum(results ** 2))
