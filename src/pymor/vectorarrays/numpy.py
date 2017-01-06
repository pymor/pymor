# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.sparse import issparse

from pymor.core import NUMPY_INDEX_QUIRK
from pymor.core.interfaces import classinstancemethod
from pymor.vectorarrays.interfaces import VectorArrayInterface, VectorSpaceInterface, _INDEXTYPES


class NumpyVectorArray(VectorArrayInterface):
    """|VectorArray| implementation via |NumPy arrays|.

    This is the default |VectorArray| type used by all |Operators|
    in pyMOR's discretization toolkit. Moreover, all reduced |Operators|
    are based on |NumpyVectorArray|.

    This class is just a thin wrapper around the underlying
    |NumPy array|. Thus, while operations like
    :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.axpy` or
    :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.dot`
    will be quite efficient, removing or appending vectors will
    be costly.

    The associated |VectorSpace| is |NumpyVectorSpace|.
    """

    def __init__(self, array, space):
        self._array = array
        self.space = space
        self._refcount = [1]
        self._len = len(array)

    @property
    def data(self):
        if self._refcount[0] > 1:
            self._deep_copy()
        return self._array[:self._len]

    @property
    def real(self):
        return NumpyVectorArray(self._array[:self._len].real.copy(), self.space)

    @property
    def imag(self):
        return NumpyVectorArray(self._array[:self._len].imag, self.space)

    def __len__(self):
        return self._len

    def __getitem__(self, ind):
        return NumpyVectorArrayView(self, ind)

    def __delitem__(self, ind):
        assert self.check_ind(ind)
        if self._refcount[0] > 1:
            self._deep_copy()

        if type(ind) is slice:
            ind = set(range(*ind.indices(self._len)))
        elif not hasattr(ind, '__len__'):
            ind = {ind if 0 <= ind else self._len + ind}
        else:
            l = self._len
            ind = set(i if 0 <= i else l+i for i in ind)
        remaining = sorted(set(range(len(self))) - ind)
        self._array = self._array[remaining]
        self._len = len(self._array)
        if not self._array.flags['OWNDATA']:
            self._array = self._array.copy()

    def copy(self, deep=False, *, _ind=None):
        if _ind is None and not deep:
            C = NumpyVectorArray(self._array, self.space)
            C._len = self._len
            C._refcount = self._refcount
            self._refcount[0] += 1
            return C
        else:
            new_array = self._array[:self._len] if _ind is None else self._array[_ind]
            if not new_array.flags['OWNDATA']:
                new_array = new_array.copy()
            return NumpyVectorArray(new_array, self.space)

    def append(self, other, remove_from_other=False):
        assert self.dim == other.dim
        assert not remove_from_other or (other is not self and getattr(other, 'base', None) is not self)

        if self._refcount[0] > 1:
            self._deep_copy()

        other_array = other.data
        len_other = len(other_array)
        if len_other == 0:
            return

        if len_other <= self._array.shape[0] - self._len:
            if self._array.dtype != other_array.dtype:
                self._array = self._array.astype(np.promote_types(self._array.dtype, other_array.dtype))
            self._array[self._len:self._len + len_other] = other_array
        else:
            self._array = np.append(self._array[:self._len], other_array, axis=0)
        self._len += len_other

        if remove_from_other:
            if other.is_view:
                del other.base[other.ind]
            else:
                del other[:]

    def scal(self, alpha, *, _ind=None):
        if _ind is None:
            _ind = slice(0, self._len)
        assert isinstance(alpha, _INDEXTYPES) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (self.len_ind(_ind),)

        if self._refcount[0] > 1:
            self._deep_copy()

        if type(alpha) is np.ndarray:
            alpha = alpha[:, np.newaxis]

        alpha_type = type(alpha)
        alpha_dtype = alpha.dtype if alpha_type is np.ndarray else alpha_type
        if self._array.dtype != alpha_dtype:
            self._array = self._array.astype(np.promote_types(self._array.dtype, alpha_dtype))
        self._array[_ind] *= alpha

    def axpy(self, alpha, x, *, _ind=None):
        if _ind is None:
            _ind = slice(0, self._len)
        assert self.dim == x.dim
        assert isinstance(alpha, _INDEXTYPES) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (self.len_ind(_ind),)

        if self._refcount[0] > 1:
            self._deep_copy()

        B = x.base._array[x.ind] if x.is_view else x._array[:x._len]
        assert self.len_ind(_ind) == len(B) or len(B) == 1

        alpha_type = type(alpha)
        alpha_dtype = alpha.dtype if alpha_type is np.ndarray else alpha_type
        if self._array.dtype != alpha_dtype or self._array.dtype != B.dtype:
            dtype = np.promote_types(self._array.dtype, alpha_dtype)
            dtype = np.promote_types(dtype, B.dtype)
            self._array = self._array.astype(dtype)

        if type(alpha) is np.ndarray:
            alpha = alpha[:, np.newaxis]
        self._array[_ind] += B * alpha

    def dot(self, other, *, _ind=None):
        if _ind is None:
            _ind = slice(0, self._len)
        assert self.dim == other.dim

        A = self._array[_ind]
        B = other.base._array[other.ind] if other.is_view else other._array[:other._len]

        if B.dtype in _complex_dtypes:
            return A.dot(B.conj().T)
        else:
            return A.dot(B.T)

    def pairwise_dot(self, other, *, _ind=None):
        if _ind is None:
            _ind = slice(0, self._len)
        assert self.dim == other.dim

        A = self._array[_ind]
        B = other.base._array[other.ind] if other.is_view else other._array[:other._len]

        assert len(A) == len(B)

        if B.dtype in _complex_dtypes:
            return np.sum(A * B.conj(), axis=1)
        else:
            return np.sum(A * B, axis=1)

    def lincomb(self, coefficients, *, _ind=None):
        if _ind is None:
            _ind = slice(0, self._len)
        assert 1 <= coefficients.ndim <= 2

        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, ...]

        return NumpyVectorArray(coefficients.dot(self._array[_ind]), self.space)

    def l1_norm(self, *, _ind=None):
        if _ind is None:
            _ind = slice(0, self._len)
        return np.linalg.norm(self._array[_ind], ord=1, axis=1)

    def l2_norm(self, *, _ind=None):
        if _ind is None:
            _ind = slice(0, self._len)
        return np.linalg.norm(self._array[_ind], axis=1)

    def l2_norm2(self, *, _ind=None):
        if _ind is None:
            _ind = slice(0, self._len)
        A = self._array[_ind]
        return np.sum((A * A.conj()).real, axis=1)

    def sup_norm(self, *, _ind=None):
        if self.dim == 0:
            if _ind is None:
                _ind = slice(0, self._len)
            return np.zeros(self.len_ind(_ind))
        else:
            _, max_val = self.amax(_ind=_ind)
            return max_val

    def components(self, component_indices, *, _ind=None):
        if _ind is None:
            _ind = slice(0, self._len)
        assert isinstance(component_indices, list) and (len(component_indices) == 0 or min(component_indices) >= 0) \
            or (isinstance(component_indices, np.ndarray) and component_indices.ndim == 1
                and (len(component_indices) == 0 or np.min(component_indices) >= 0))
        # NumPy 1.9 is quite permissive when indexing arrays of size 0, so we have to add the
        # following check:
        assert self._len > 0 \
            or (isinstance(component_indices, list)
                and (len(component_indices) == 0 or max(component_indices) < self.dim)) \
            or (isinstance(component_indices, np.ndarray) and component_indices.ndim == 1
                and (len(component_indices) == 0 or np.max(component_indices) < self.dim))

        if NUMPY_INDEX_QUIRK and (self._len == 0 or self.dim == 0):
            assert isinstance(component_indices, list) \
                and (len(component_indices) == 0 or max(component_indices) < self.dim) \
                or isinstance(component_indices, np.ndarray) \
                and component_indices.ndim == 1 \
                and (len(component_indices) == 0 or np.max(component_indices) < self.dim)
            return np.zeros((self.len_ind(_ind), len(component_indices)))

        return self._array[:, component_indices][_ind, :]

    def amax(self, *, _ind=None):
        if _ind is None:
            _ind = slice(0, self._len)
        assert self.dim > 0

        if self._array.shape[1] == 0:
            l = self.len_ind(_ind)
            return np.ones(l) * -1, np.zeros(l)

        A = np.abs(self._array[_ind])
        max_ind = np.argmax(A, axis=1)
        max_val = A[np.arange(len(A)), max_ind]
        return max_ind, max_val

    def __str__(self):
        return self._array[:self._len].__str__()

    def __repr__(self):
        return 'NumpyVectorArray({}, {})'.format(self._array[:self._len].__str__(), self.space)

    def __del__(self):
        self._refcount[0] -= 1

    def _deep_copy(self):
        self._array = self._array.copy()  # copy the array data
        self._refcount[0] -= 1            # decrease refcount for original array
        self._refcount = [1]              # create new reference counter

    def __add__(self, other):
        if isinstance(other, _INDEXTYPES):
            assert other == 0
            return self.copy()
        assert self.dim == other.dim
        return NumpyVectorArray(self._array[:self._len] +
                                (other.base._array[other.ind] if other.is_view else other._array[:other._len]),
                                self.space)

    def __iadd__(self, other):
        assert self.dim == other.dim
        if self._refcount[0] > 1:
            self._deep_copy()
        self._array[:self._len] += other.base._array[other.ind] if other.is_view else other._array[:other._len]
        return self

    __radd__ = __add__

    def __sub__(self, other):
        assert self.dim == other.dim
        return NumpyVectorArray(self._array[:self._len] -
                                (other.base._array[other.ind] if other.is_view else other._array[:other._len]),
                                self.space)

    def __isub__(self, other):
        assert self.dim == other.dim
        if self._refcount[0] > 1:
            self._deep_copy()
        self._array[:self._len] -= other.base._array[other.ind] if other.is_view else other._array[:other._len]
        return self

    def __mul__(self, other):
        assert isinstance(other, _INDEXTYPES) \
            or isinstance(other, np.ndarray) and other.shape == (len(self),)
        return NumpyVectorArray(self._array[:self._len] * other, self.space)

    def __imul__(self, other):
        assert isinstance(other, _INDEXTYPES) \
            or isinstance(other, np.ndarray) and other.shape == (len(self),)
        if self._refcount[0] > 1:
            self._deep_copy()
        self._array[:self._len] *= other
        return self

    def __neg__(self):
        return NumpyVectorArray(-self._array[:self._len], self.space)


class NumpyVectorSpace(VectorSpaceInterface):
    """|VectorSpace| of |NumpyVectorArrays|.

    Parameters
    ----------
    dim
        The dimension of the vectors contained in the space.
    id
        See :attr:`~pymor.vectorarrays.interfaces.VectorSpaceInterface.id`.
    """

    def __init__(self, dim, id_=None):
        self.dim = dim
        self.id = id_

    def __eq__(self, other):
        return type(other) is type(self) and self.dim == other.dim and self.id == other.id

    def zeros(self, count=1, reserve=0):
        assert count >= 0
        assert reserve >= 0
        va = NumpyVectorArray(np.empty((0, 0)), self)
        va._array = np.zeros((max(count, reserve), self.dim))
        va._len = count
        return va

    @classinstancemethod
    def make_array(cls, obj, id_=None):
        return cls._array_factory(obj, id_=id_)

    @make_array.instancemethod
    def make_array(self, obj):
        return self._array_factory(obj, space=self)

    @classinstancemethod
    def from_data(cls, data, id_=None):
        return cls._array_factory(data, id_=id_)

    @from_data.instancemethod
    def from_data(self, data):
        return self._array_factory(data, space=self)

    @classinstancemethod
    def from_file(cls, path, key=None, single_vector=False, transpose=False, id_=None):
        assert not (single_vector and transpose)
        from pymor.tools.io import load_matrix
        array = load_matrix(path, key=key)
        assert isinstance(array, np.ndarray)
        assert array.ndim <= 2
        if array.ndim == 1:
            array = array.reshape((1, -1))
        if single_vector:
            assert array.shape[0] == 1 or array.shape[1] == 1
            array = array.reshape((1, -1))
        if transpose:
            array = array.T
        return cls.make_array(array, id_=id_)

    @from_file.instancemethod
    def from_file(self, path, key=None, single_vector=False, transpose=False):
        return self.from_file(path, key=key, single_vector=single_vector, transpose=transpose, id_=self.id)

    @classmethod
    def _array_factory(cls, array, space=None, id_=None):
        if type(array) is np.ndarray:
            pass
        elif issparse(array):
            array = array.toarray()
        elif hasattr(array, 'data'):
            array = array.data
        else:
            array = np.array(array, ndmin=2)
        if array.ndim != 2:
            assert array.ndim == 1
            array = np.reshape(array, (1, -1))
        if space is None:
            return NumpyVectorArray(array, cls(array.shape[1], id_))
        else:
            assert array.shape[1] == space.dim
            return NumpyVectorArray(array, space)

    @property
    def is_scalar(self):
        return self.dim == 1

    def __repr__(self):
        return 'NumpyVectorSpace({})'.format(self.dim) if self.id is None \
            else 'NumpyVectorSpace({}, {})'.format(self.dim, self.id)


class NumpyVectorArrayView(NumpyVectorArray):

    is_view = True

    def __init__(self, array, ind):
        assert array.check_ind(ind)
        self.base = array
        self.ind = array.normalize_ind(ind)
        self.space = array.space

    @property
    def data(self):
        return self.base.data[self.ind]

    def __len__(self):
        return self.base.len_ind(self.ind)

    def __getitem__(self, ind):
        return self.base[self.base.sub_index(self.ind, ind)]

    def __delitem__(self):
        raise ValueError('Cannot remove from NumpyVectorArrayView')

    def append(self, other, remove_from_other=False):
        raise ValueError('Cannot append to NumpyVectorArrayView')

    def copy(self, deep=False):
        return self.base.copy(_ind=self.ind, deep=deep)

    def scal(self, alpha):
        assert self.base.check_ind_unique(self.ind)
        self.base.scal(alpha, _ind=self.ind)

    def axpy(self, alpha, x):
        assert self.base.check_ind_unique(self.ind)
        self.base.axpy(alpha, x, _ind=self.ind)

    def dot(self, other):
        return self.base.dot(other, _ind=self.ind)

    def pairwise_dot(self, other):
        return self.base.pairwise_dot(other, _ind=self.ind)

    def lincomb(self, coefficients):
        return self.base.lincomb(coefficients, _ind=self.ind)

    def l1_norm(self):
        return self.base.l1_norm(_ind=self.ind)

    def l2_norm(self):
        return self.base.l2_norm(_ind=self.ind)

    def l2_norm2(self):
        return self.base.l2_norm2(_ind=self.ind)
        pass

    def sup_norm(self):
        return self.base.sup_norm(_ind=self.ind)

    def components(self, component_indices):
        return self.base.components(component_indices, _ind=self.ind)

    def amax(self):
        return self.base.amax(_ind=self.ind)

    def __add__(self, other):
        if isinstance(other, _INDEXTYPES):
            assert other == 0
            return self.copy()
        assert self.dim == other.dim
        return NumpyVectorArray(self.base._array[self.ind] +
                                (other.base._array[other.ind] if other.is_view else other._array[:other._len]),
                                self.space)

    def __iadd__(self, other):
        assert self.dim == other.dim
        assert self.base.check_ind_unique(self.ind)
        if self.base._refcount[0] > 1:
            self._deep_copy()
        self.base.array[self.ind] += other.base._array[other.ind] if other.is_view else other._array[:other._len]
        return self

    __radd__ = __add__

    def __sub__(self, other):
        assert self.dim == other.dim
        return NumpyVectorArray(self.base._array[self.ind] -
                                (other.base._array[other.ind] if other.is_view else other._array[:other._len]),
                                self.space)

    def __isub__(self, other):
        assert self.dim == other.dim
        assert self.base.check_ind_unique(self.ind)
        if self.base._refcount[0] > 1:
            self._deep_copy()
        self.base._array[self.ind] -= other.base._array[other.ind] if other.is_view else other._array[:other._len]
        return self

    def __mul__(self, other):
        assert isinstance(other, _INDEXTYPES) \
            or isinstance(other, np.ndarray) and other.shape == (len(self),)
        return NumpyVectorArray(self.base._array[self.ind] * other, self.space)

    def __imul__(self, other):
        assert isinstance(other, _INDEXTYPES) \
            or isinstance(other, np.ndarray) and other.shape == (len(self),)
        assert self.base.check_ind_unique(self.ind)
        if self.base._refcount[0] > 1:
            self._deep_copy()
        self.base._array[self.ind] *= other
        return self

    def __neg__(self):
        return NumpyVectorArray(-self.base._array[self.ind], self.space)

    def __del__(self):
        return

    def __repr__(self):
        return 'NumpyVectorArrayView({}, {})'.format(self.data, self.space)


_complex_dtypes = (np.complex64, np.complex128)
