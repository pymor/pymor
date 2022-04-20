# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Wrapper classes for building MPI distributed |VectorArrays|.

This module contains several wrapper classes which allow to
transform single rank |VectorArrays| into MPI distributed
|VectorArrays| which can be used on rank 0 like ordinary
|VectorArrays|.

The implementations are based on the event loop provided
by :mod:`pymor.tools.mpi`.
"""

import numpy as np

from pymor.core.pickle import unpicklable
from pymor.tools import mpi
from pymor.vectorarrays.interface import VectorArray, VectorArrayImpl, VectorSpace


class MPIVectorArrayImpl(VectorArrayImpl):

    def to_numpy(self, ensure_copy, ind):
        raise NotImplementedError

    def __init__(self, obj_id, space):
        self.obj_id = obj_id
        self.space = space

    def __len__(self):
        return mpi.call(mpi.method_call, self.obj_id, '__len__')

    def delete(self, ind):
        mpi.call(mpi.method_call, self.obj_id, '__delitem__', ind)

    def copy(self, deep, ind):
        return type(self)(mpi.call(mpi.function_call_manage, _MPIVectorArray_copy, self.obj_id, deep, ind),
                          self.space)

    def append(self, other, remove_from_other, oind):
        mpi.call(mpi.function_call, _MPIVectorArray_append, self.obj_id, other.obj_id, remove_from_other, oind)

    def scal(self, alpha, ind):
        mpi.call(mpi.function_call, _MPIVectorArray_scal, self.obj_id, alpha, ind)

    def axpy(self, alpha, x, ind, xind):
        mpi.call(mpi.function_call_manage, _MPIVectorArray_axpy, self.obj_id, alpha, x.obj_id, ind, xind)

    def inner(self, other, ind, oind):
        return mpi.call(mpi.function_call, _MPIVectorArray_inner, self.obj_id, other.obj_id, ind, oind)

    def pairwise_inner(self, other, ind, oind):
        return mpi.call(mpi.function_call, _MPIVectorArray_pairwise_inner, self.obj_id, other.obj_id, ind, oind)

    def lincomb(self, coefficients, ind):
        return type(self)(mpi.call(mpi.function_call_manage, _MPIVectorArray_lincomb, self.obj_id, coefficients, ind),
                          self.space)

    def norm2(self, ind):
        return mpi.call(mpi.function_call, _MPIVectorArray_norm2, self.obj_id, ind)

    def dofs(self, dof_indices, ind):
        return mpi.call(mpi.function_call, _MPIVectorArray_dofs, self.obj_id, dof_indices, ind)

    def amax(self, ind):
        return mpi.call(mpi.function_call, _MPIVectorArray_amax, self.obj_id, ind)

    def __del__(self):
        mpi.call(mpi.remove_object, self.obj_id)

    def real(self, ind):
        return type(self)(mpi.call(mpi.function_call_manage, _MPIVectorArray_real, self.obj_id, ind), self.space)

    def imag(self, ind):
        return type(self)(mpi.call(mpi.function_call_manage, _MPIVectorArray_imag, self.obj_id, ind), self.space)

    def conj(self, ind):
        return type(self)(mpi.call(mpi.function_call_manage, _MPIVectorArray_conj, self.obj_id, ind), self.space)


@unpicklable
class MPIVectorArray(VectorArray):
    """MPI distributed |VectorArray|.

    Given a local |VectorArray| on each MPI rank, this wrapper class
    uses the event loop from :mod:`pymor.tools.mpi` to build a global
    MPI distributed vector array from these local arrays.

    Instances of `MPIVectorArray` can be used on rank 0 like any
    other (non-distributed) |VectorArray|.

    Note, however, that the implementation of the local VectorArrays
    needs to be MPI aware. For instance, `cls.inner` must perform the
    needed MPI communication to sum up the local inner products and
    return the sums on rank 0.

    Default implementations for all communication requiring
    interface methods are provided by :class:`MPIVectorArrayAutoComm`
    (also see :class:`MPIVectorArrayNoComm`).

    Note that resource cleanup is handled by :meth:`object.__del__`.
    Please be aware of the peculiarities of destructors in Python!

    The associated |VectorSpace| is :class:`MPIVectorSpace`.
    """

    impl_type = MPIVectorArrayImpl


class MPIVectorSpace(VectorSpace):
    """|VectorSpace| of :class:`MPIVectorArrays <MPIVectorArray>`.

    Parameters
    ----------
    local_spaces
        `tuple` of the different |VectorSpaces| of the local
        |VectorArrays| on the MPI ranks.
        Alternatively, the length of `local_spaces` may be 1, in which
        case the same |VectorSpace| is assumed for all ranks.
    """

    array_type = MPIVectorArray

    def __init__(self, local_spaces):
        self.local_spaces = tuple(local_spaces)
        if type(local_spaces[0]) is RegisteredLocalSpace:
            self.id = _local_space_registry[local_spaces[0]].id
        else:
            self.id = local_spaces[0].id

    def make_array(self, obj_id):
        """Create array from rank-local |VectorArray| instances.

        Parameters
        ----------
        obj_id
            :class:`~pymor.tools.mpi.ObjectId` of the MPI distributed
            instances of `cls` wrapped by this array.

        Returns
        -------
        The newly created :class:`MPIVectorArray`.
        """
        assert mpi.call(_MPIVectorSpace_check_local_spaces,
                        self.local_spaces, obj_id)
        return self.array_type(self, self.array_type.impl_type(obj_id, self))

    def zeros(self, count=1, reserve=0):
        return self.array_type(
            self,
            self.array_type.impl_type(
                mpi.call(_MPIVectorSpace_zeros, self.local_spaces, count=count, reserve=reserve),
                self
            )
        )

    @property
    def dim(self):
        return mpi.call(_MPIVectorSpace_dim, self.local_spaces)

    def __eq__(self, other):
        return type(other) is MPIVectorSpace and \
            len(self.local_spaces) == len(other.local_spaces) and \
            all(ls == ols for ls, ols in zip(self.local_spaces, other.local_spaces))

    def __repr__(self):
        return f'{self.__class__}({self.local_spaces}, {self.id})'


class RegisteredLocalSpace(int):

    def __repr__(self):
        return f'{_local_space_registry[self]} (id: {int(self)})'


_local_space_registry = {}
_local_space_to_id = {}


def _register_local_space(local_space):
    local_space_id = _local_space_to_id.get(local_space)
    if local_space_id is None:
        local_space_id = RegisteredLocalSpace(len(_local_space_registry))
        _local_space_registry[local_space_id] = local_space
        _local_space_to_id[local_space] = local_space_id
    return local_space_id


def _get_local_space(local_spaces):
    local_space = local_spaces[mpi.rank] if len(local_spaces) > 1 else local_spaces[0]
    if type(local_space) is RegisteredLocalSpace:
        local_space = _local_space_registry[local_space]
    return local_space


def _MPIVectorSpace_zeros(local_spaces=(None,), count=0, reserve=0):
    local_space = _get_local_space(local_spaces)
    obj = local_space.zeros(count=count, reserve=reserve)
    return mpi.manage_object(obj)


def _MPIVectorSpace_dim(local_spaces):
    local_space = _get_local_space(local_spaces)
    return local_space.dim


def _MPIVectorSpace_check_local_spaces(local_spaces, obj_id):
    U = mpi.get_object(obj_id)
    local_space = _get_local_space(local_spaces)
    results = mpi.comm.gather(U in local_space, root=0)
    if mpi.rank0:
        return np.all(results)


def _MPIVectorArray_append(self, other, remove_from_other, oind):
    self.append(_indexed(other, oind), remove_from_other)


def _MPIVectorArray_copy(self, deep, ind):
    return _indexed(self, ind).copy(deep)


def _MPIVectorArray_scal(self, alpha, ind):
    _indexed(self, ind).scal(alpha)


def _MPIVectorArray_axpy(self, alpha, x, ind, xind):
    _indexed(self, ind).axpy(alpha, _indexed(x, xind))


def _MPIVectorArray_inner(self, other, ind, oind):
    return _indexed(self, ind).inner(_indexed(other, oind))


def _MPIVectorArray_pairwise_inner(self, other, ind, oind):
    return _indexed(self, ind).pairwise_inner(_indexed(other, oind))


def _MPIVectorArray_lincomb(self, coefficients, ind):
    return _indexed(self, ind).lincomb(coefficients)


def _MPIVectorArray_norm(self, ind):
    return _indexed(self, ind).norm()


def _MPIVectorArray_norm2(self, ind):
    return _indexed(self, ind).norm2()


def _MPIVectorArray_dofs(self, dof_indices, ind):
    return _indexed(self, ind).dofs(dof_indices)


def _MPIVectorArray_amax(self, ind):
    return _indexed(self, ind).amax()


def _MPIVectorArray_real(self, ind):
    return _indexed(self, ind).real


def _MPIVectorArray_imag(self, ind):
    return _indexed(self, ind).imag


def _MPIVectorArray_conj(self, ind):
    return _indexed(self, ind).conj()


class MPIVectorArrayNoCommImpl(MPIVectorArrayImpl):

    def inner(self, other, ind, oind):
        raise NotImplementedError

    def pairwise_inner(self, other, ind, oind):
        raise NotImplementedError

    def norm(self, ind):
        raise NotImplementedError

    def norm2(self, ind):
        raise NotImplementedError

    def dofs(self, dof_indices, ind):
        raise NotImplementedError

    def amax(self, ind):
        raise NotImplementedError


class MPIVectorArrayNoComm(MPIVectorArray):
    """MPI distributed |VectorArray|.

    This is a subclass of :class:`MPIVectorArray` which
    overrides all communication requiring interface methods
    to raise `NotImplementedError`.

    This is mainly useful as a security measure when wrapping
    arrays for which simply calling the respective method
    on the wrapped arrays would lead to wrong results and
    :class:`MPIVectorArrayAutoComm` cannot be used either
    (for instance in the presence of shared DOFs).

    The associated |VectorSpace| is :class:`MPIVectorSpaceNoComm`.
    """

    impl_type = MPIVectorArrayNoCommImpl


class MPIVectorSpaceNoComm(MPIVectorSpace):
    """|VectorSpace| for :class:`MPIVectorArrayNoComm`."""

    array_type = MPIVectorArrayNoComm

    @property
    def dim(self):
        raise NotImplementedError


class MPIVectorArrayAutoCommImpl(MPIVectorArrayImpl):

    def inner(self, other, ind, oind):
        return mpi.call(mpi.function_call, _MPIVectorArrayAutoComm_inner, self.obj_id, other.obj_id, ind, oind)

    def pairwise_inner(self, other, ind, oind):
        return mpi.call(mpi.function_call, _MPIVectorArrayAutoComm_pairwise_inner, self.obj_id, other.obj_id, ind, oind)

    def norm(self, ind):
        return np.sqrt(self.norm2(ind))

    def norm2(self, ind):
        return mpi.call(mpi.function_call, _MPIVectorArrayAutoComm_norm2, self.obj_id, ind)

    def dofs(self, dof_indices, ind):
        offsets = getattr(self, '_offsets', None)
        if offsets is None:
            offsets = self.space._get_dims()[1]
        dof_indices = np.array(dof_indices)
        return mpi.call(mpi.function_call, _MPIVectorArrayAutoComm_dofs, self.obj_id, offsets, dof_indices, ind)

    def amax(self, ind):
        offsets = getattr(self, '_offsets', None)
        if offsets is None:
            offsets = self.space._get_dims()[1]
        inds, vals = mpi.call(mpi.function_call, _MPIVectorArrayAutoComm_amax, self.obj_id, ind)
        inds += offsets[:, np.newaxis]
        max_inds = np.argmax(vals, axis=0)
        # np.choose does not work due to
        # https://github.com/numpy/numpy/issues/3259
        return (np.array([inds[max_inds[i], i] for i in range(len(max_inds))]),
                np.array([vals[max_inds[i], i] for i in range(len(max_inds))]))


class MPIVectorArrayAutoComm(MPIVectorArray):
    """MPI distributed |VectorArray|.

    This is a subclass of :class:`MPIVectorArray` which
    provides default implementations for all communication
    requiring interface methods for the case when the
    wrapped array is not MPI aware.

    Note, however, that depending on the model
    these default implementations might lead to wrong results
    (for instance in the presence of shared DOFs).

    The associated |VectorSpace| is :class:`MPIVectorSpaceAutoComm`.
    """

    impl_type = MPIVectorArrayAutoCommImpl


class MPIVectorSpaceAutoComm(MPIVectorSpace):
    """|VectorSpace| for :class:`MPIVectorArrayAutoComm`."""

    array_type = MPIVectorArrayAutoComm

    @property
    def dim(self):
        dim = getattr(self, '_dim', None)
        if dim is None:
            dim = self._get_dims()[0]
        return dim

    def _get_dims(self):
        dims = mpi.call(_MPIVectorSpaceAutoComm_dim, self.local_spaces)
        self._offsets = offsets = np.cumsum(np.concatenate(([0], dims)))[:-1]
        self._dim = dim = sum(dims)
        return dim, offsets


def _MPIVectorSpaceAutoComm_dim(local_spaces):
    local_space = _get_local_space(local_spaces)
    dims = mpi.comm.gather(local_space, root=0)
    if mpi.rank0:
        return dims


def _MPIVectorArrayAutoComm_inner(self, other, ind, oind):
    local_results = _indexed(self, ind).inner(_indexed(other, oind))
    assert local_results.dtype == np.float64
    results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_results, results, root=0)
    if mpi.rank0:
        return np.sum(results, axis=0)


def _MPIVectorArrayAutoComm_pairwise_inner(self, other, ind, oind):
    local_results = _indexed(self, ind).pairwise_inner(_indexed(other, oind))
    assert local_results.dtype == np.float64
    results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_results, results, root=0)
    if mpi.rank0:
        return np.sum(results, axis=0)


def _MPIVectorArrayAutoComm_norm2(self, ind):
    local_results = _indexed(self, ind).norm2()
    assert local_results.dtype == np.float64
    results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_results, results, root=0)
    if mpi.rank0:
        return np.sum(results, axis=0)


def _MPIVectorArrayAutoComm_dofs(self, offsets, dof_indices, ind):
    self = _indexed(self, ind)
    offset = offsets[mpi.rank]
    dim = self.dim
    my_indices = np.logical_and(dof_indices >= offset, dof_indices < offset + dim)
    local_results = np.zeros((len(self), len(dof_indices)))
    local_results[:, my_indices] = self.dofs(dof_indices[my_indices] - offset)
    assert local_results.dtype == np.float64
    results = np.empty((mpi.size,) + local_results.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_results, results, root=0)
    if mpi.rank0:
        return np.sum(results, axis=0)


def _MPIVectorArrayAutoComm_amax(self, ind):
    self = _indexed(self, ind)
    local_inds, local_vals = self.amax()
    assert local_inds.dtype == np.int64
    assert local_vals.dtype == np.float64
    inds = np.empty((mpi.size,) + local_inds.shape, dtype=np.int64) if mpi.rank0 else None
    vals = np.empty((mpi.size,) + local_inds.shape, dtype=np.float64) if mpi.rank0 else None
    mpi.comm.Gather(local_inds, inds, root=0)
    mpi.comm.Gather(local_vals, vals, root=0)
    if mpi.rank0:
        return inds, vals


def _indexed(array, ind):
    return array if ind is None else array[ind]
