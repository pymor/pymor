# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Wrapper classes for building MPI distributed |VectorArrays|.

This module contains several wrapper classes which allow to
transform single rank |VectorArrays| into MPI distributed
|VectorArrays| which can be used on rank 0 like ordinary
|VectorArrays|. Similar classes are provided for handling
:class:`Vectors <pymor.vectorarrays.list.VectorInterface>`.
The implementations are based on the event loop provided
by :mod:`pymor.tools.mpi`.
"""

import numpy as np

from pymor.tools import mpi
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.list import VectorInterface


class MPIVectorArray(VectorArrayInterface):
    """MPI distributed VectorArray.

    Given a single-rank |VectorArray| implementation `cls`, this
    wrapper class uses the event loop from :mod:`pymor.tools.mpi`
    to build MPI distributed vector arrays where on each MPI rank
    an instance of `cls` is used to manage the local data.

    Instances of `MPIVectorArray` can be used on rank 0 like any
    other (non-distributed) |VectorArray|.

    Note, however, that the implementation of `cls` needs to be
    MPI aware. For instance, `cls.dot` must perform the needed
    MPI communication to sum up the local scalar products and
    return the sums on rank 0.

    Default implementations for all communication requiring
    interface methods are provided by :class:`MPIVectorArrayAutoComm`.
    See also :class:`MPIVectorArrayNoComm`.

    Note that resource cleanup is handled by :meth:`object.__del__`.
    Please be aware of the peculiarities of destructors in Python!

    Parameters
    ----------
    cls
        The class of the |VectorArray| implementation used on
        every MPI rank.
    subtype
        `tuple` of the different subtypes of `cls` on the MPI ranks.
        Alternatively, the length of subtype may be 1 in which
        case the same subtype is assumed for all ranks.
    obj_id
        :class:`~pymor.tools.mpi.ObjectId` of the MPI distributed
        instances of `cls` wrapped by this array.
    """

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

    def copy(self, ind=None, deep=False):
        return type(self)(self.cls, self.array_subtype,
                          mpi.call(mpi.method_call_manage, self.obj_id, 'copy', ind=ind, deep=deep))

    def append(self, other, o_ind=None, remove_from_other=False):
        mpi.call(mpi.method_call, self.obj_id, 'append', other.obj_id,
                 o_ind=o_ind, remove_from_other=remove_from_other)

    def remove(self, ind=None):
        mpi.call(mpi.method_call, self.obj_id, 'remove', ind=ind)

    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        mpi.call(mpi.method_call, self.obj_id, 'replace', other.obj_id,
                 ind=ind, o_ind=o_ind, remove_from_other=remove_from_other)

    def scal(self, alpha, ind=None):
        mpi.call(mpi.method_call, self.obj_id, 'scal', alpha, ind=ind)

    def axpy(self, alpha, x, ind=None, x_ind=None):
        mpi.call(_MPIVectorArray_axpy, self.obj_id, alpha, x.obj_id, ind=ind, x_ind=x_ind)

    def dot(self, other, ind=None, o_ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'dot', other.obj_id, ind=ind, o_ind=o_ind)

    def pairwise_dot(self, other, ind=None, o_ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'pairwise_dot', other.obj_id, ind=ind, o_ind=o_ind)

    def lincomb(self, coefficients, ind=None):
        return type(self)(self.cls, self.array_subtype,
                          mpi.call(mpi.method_call_manage, self.obj_id, 'lincomb', coefficients, ind=ind))

    def l1_norm(self, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'l1_norm', ind=ind)

    def l2_norm(self, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'l2_norm', ind=ind)

    def l2_norm2(self, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'l2_norm2', ind=ind)

    def components(self, component_indices, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'components', component_indices, ind=ind)

    def amax(self, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'amax', ind=ind)

    def __del__(self):
        mpi.call(mpi.remove_object, self.obj_id)


class RegisteredSubtype(int):
    pass


_subtype_registry = {}
_subtype_to_id = {}


def _register_subtype(subtype):
    # if mpi.rank == 0:
    #     import pdb; pdb.set_trace()
    subtype_id = _subtype_to_id.get(subtype)
    if subtype_id is None:
        subtype_id = RegisteredSubtype(len(_subtype_registry))
        _subtype_registry[subtype_id] = subtype
        _subtype_to_id[subtype] = subtype_id
    return subtype_id


def _MPIVectorArray_make_array(cls, subtype=(None,), count=0, reserve=0):
    subtype = subtype[mpi.rank] if len(subtype) > 1 else subtype[0]
    if type(subtype) is RegisteredSubtype:
        subtype = _subtype_registry[subtype]
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
    """MPI distributed VectorArray.

    This is a subclass of :class:`MPIVectorArray` which
    overrides all communication requiring interface methods
    to raise `NotImplementedError`.

    This is mainly useful as a security measure when wrapping
    arrays for which simply calling the respective method
    on the wrapped arrays would lead to wrong results and
    :class:`MPIVectorArrayAutoComm` cannot be used either
    (for instance in the presence of shared DOFs).
    """

    @property
    def dim(self):
        raise NotImplementedError

    def dot(self, other, ind=None, o_ind=None):
        raise NotImplementedError

    def pairwise_dot(self, other, ind=None, o_ind=None):
        raise NotImplementedError

    def l1_norm(self, ind=None):
        raise NotImplementedError

    def l2_norm(self, ind=None):
        raise NotImplementedError

    def l2_norm2(self, ind=None):
        raise NotImplementedError

    def components(self, component_indices, ind=None):
        raise NotImplementedError

    def amax(self, ind=None):
        raise NotImplementedError


class MPIVectorArrayAutoComm(MPIVectorArray):
    """MPI distributed VectorArray.

    This is a subclass of :class:`MPIVectorArray` which
    provides default implementations for all communication
    requiring interface methods for the case when the
    wrapped array is not MPI aware.

    Note, however, that depending on the discretization
    these default implementations might lead to wrong results
    (for instance in the presence of shared DOFs).
    """

    @property
    def dim(self):
        dim = getattr(self, '_dim', None)
        if dim is None:
            dim = self._get_dims()[0]
        return dim

    def dot(self, other, ind=None, o_ind=None):
        return mpi.call(_MPIVectorArrayAutoComm_dot, self.obj_id, other.obj_id, ind=ind, o_ind=o_ind)

    def pairwise_dot(self, other, ind=None, o_ind=None):
        return mpi.call(_MPIVectorArrayAutoComm_pairwise_dot, self.obj_id, other.obj_id, ind=ind, o_ind=o_ind)

    def l1_norm(self, ind=None):
        return mpi.call(_MPIVectorArrayAutoComm_l1_norm, self.obj_id, ind=ind)

    def l2_norm(self, ind=None):
        return mpi.call(_MPIVectorArrayAutoComm_l2_norm, self.obj_id, ind=ind)

    def l2_norm2(self, ind=None):
        return mpi.call(_MPIVectorArrayAutoComm_l2_norm2, self.obj_id, ind=ind)

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
        # np.choose does not work due to
        # https://github.com/numpy/numpy/issues/3259
        return (np.array([inds[max_inds[i], i] for i in range(len(max_inds))]),
                np.array([vals[max_inds[i], i] for i in range(len(max_inds))]))

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
    local_results = self.l2_norm2(ind=ind)
    assert local_results.dtype == np.float64
    results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_results, results, root=0)
    if mpi.rank0:
        return np.sqrt(np.sum(results, axis=0))


def _MPIVectorArrayAutoComm_l2_norm2(self, ind=None):
    self = mpi.get_object(self)
    local_results = self.l2_norm2(ind=ind)
    assert local_results.dtype == np.float64
    results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_results, results, root=0)
    if mpi.rank0:
        return np.sum(results, axis=0)


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
    """MPI distributed Vector.

    Given a single-rank implemenation of
    :class:`~pymor.vectorarrays.list.VectorInterface` `cls`, this
    wrapper class uses the event loop from :mod:`pymor.tools.mpi`
    to build MPI distributed vector where on each MPI rank
    an instance of `cls` is used to manage the local data.

    Instances of `MPIVector` can be used on rank 0 in conjunction
    with |ListVectorArray| like any other (non-distributed) Vector
    class.

    Note, however, that the implementation of `cls` needs to be
    MPI aware. For instance, `cls.dot` must perform the needed
    MPI communication to sum up the local scalar products and
    return the sum on rank 0.

    Default implementations for all communication requiring
    interface methods are provided by :class:`MPIVectorAutoComm`.
    See also :class:`MPIVectorNoComm`.

    Note that resource cleanup is handled by :meth:`object.__del__`.
    Please be aware of the peculiarities of destructors in Python!

    Parameters
    ----------
    cls
        The class of the :class:`~pymor.vectorarrays.list.VectorInterface`
        implementation used on every MPI rank.
    subtype
        `tuple` of the different subtypes of `cls` on the MPI ranks.
        Alternatively, the length of subtype may be 1 in which
        case the same subtype is assumed for all ranks.
    obj_id
        :class:`~pymor.tools.mpi.ObjectId` of the MPI distributed
        instances of `cls` wrapped by this object.
    """

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

    def copy(self, deep=False):
        return type(self)(self.cls, self.vector_subtype,
                          mpi.call(mpi.method_call_manage, self.obj_id, 'copy', deep=deep))

    def scal(self, alpha):
        mpi.call(mpi.method_call, self.obj_id, 'scal', alpha)

    def axpy(self, alpha, x):
        mpi.call(_MPIVector_axpy, self.obj_id, alpha, x.obj_id)

    def dot(self, other):
        return mpi.call(mpi.method_call, self.obj_id, 'dot', other.obj_id)

    def l1_norm(self):
        return mpi.call(mpi.method_call, self.obj_id, 'l1_norm')

    def l2_norm(self):
        return mpi.call(mpi.method_call, self.obj_id, 'l2_norm')

    def l2_norm2(self):
        return mpi.call(mpi.method_call, self.obj_id, 'l2_norm2')

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
    if type(subtype) is RegisteredSubtype:
        subtype = _subtype_registry[subtype]
    obj = cls.make_zeros(subtype)
    return mpi.manage_object(obj)


class MPIVectorNoComm(object):
    """MPI distributed Vector.

    This is a subclass of :class:`MPIVector` which
    overrides all communication requiring interface methods
    to raise `NotImplementedError`.

    This is mainly useful as a security measure when wrapping
    vectors for which simply calling the respective method
    on the wrapped vectors would lead to wrong results and
    :class:`MPIVectorAutoComm` cannot be used either
    (for instance in the presence of shared DOFs).
    """

    @property
    def dim(self):
        raise NotImplementedError

    def dot(self, other):
        raise NotImplementedError

    def pairwise_dot(self, other):
        raise NotImplementedError

    def l1_norm(self):
        raise NotImplementedError

    def l2_norm(self):
        raise NotImplementedError

    def l2_norm2(self):
        raise NotImplementedError

    def components(self, component_indices):
        raise NotImplementedError

    def amax(self):
        raise NotImplementedError


class MPIVectorAutoComm(MPIVector):
    """MPI distributed Vector.

    This is a subclass of :class:`MPIArray` which
    provides default implementations for all communication
    requiring interface methods for the case when the
    wrapped vector is not MPI aware.

    Note, however, that depending on the discretization
    these default implementations might lead to wrong results
    (for instance in the presence of shared DOFs).
    """

    @property
    def dim(self):
        return mpi.call(_MPIVectorAutoComm_dim, self.obj_id)

    def dot(self, other):
        return mpi.call(_MPIVectorAutoComm_dot, self.obj_id, other.obj_id)

    def l1_norm(self):
        return mpi.call(_MPIVectorAutoComm_l1_norm, self.obj_id)

    def l2_norm(self):
        return mpi.call(_MPIVectorAutoComm_l2_norm, self.obj_id)

    def l2_norm2(self):
        return mpi.call(_MPIVectorAutoComm_l2_norm2, self.obj_id)

    def components(self, component_indices):
        raise NotImplementedError

    def amax(self):
        raise NotImplementedError


def _MPIVectorAutoComm_dim(self):
    self = mpi.get_object(self)
    dims = mpi.comm.gather(self.dim, root=0)
    if mpi.rank0:
        return sum(dims)


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
    local_result = self.l2_norm2()
    assert local_result.dtype == np.float64
    results = np.empty((mpi.size,), dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_result, results, root=0)
    if mpi.rank0:
        return np.sqrt(np.sum(results))


def _MPIVectorAutoComm_l2_norm2(self):
    self = mpi.get_object(self)
    local_result = self.l2_norm2()
    assert local_result.dtype == np.float64
    results = np.empty((mpi.size,), dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_result, results, root=0)
    if mpi.rank0:
        return np.sum(results)
