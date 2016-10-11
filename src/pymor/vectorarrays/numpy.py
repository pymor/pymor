# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.sparse import issparse

from pymor.core import NUMPY_INDEX_QUIRK
from pymor.vectorarrays.interfaces import VectorArrayInterface, VectorSpace, _INDEXTYPES


class NumpyVectorArray(VectorArrayInterface):
    """|VectorArray| implementation via |NumPy arrays|.

    This is the default |VectorArray| type used by all |Operators|
    in pyMOR's discretization toolkit. Moreover, all reduced |Operators|
    are based on |NumpyVectorArray|.

    Note that this class is just a thin wrapper around the underlying
    |NumPy array|. Thus, while operations like
    :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.axpy` or
    :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.dot`
    will be quite efficient, removing or appending vectors will
    be costly.
    """

    def __init__(self, instance, dtype=None, copy=False, order=None, subok=False):
        assert not isinstance(instance, np.matrixlib.defmatrix.matrix)
        if isinstance(instance, np.ndarray):
            if copy:
                self._array = instance.copy()
            else:
                self._array = instance
        elif issparse(instance):
            self._array = instance.toarray()
        elif hasattr(instance, 'data'):
            self._array = instance.data
            if copy:
                self._array = self._array.copy()
        else:
            self._array = np.array(instance, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=2)
        if self._array.ndim != 2:
            assert self._array.ndim == 1
            self._array = np.reshape(self._array, (1, -1))
        self._len = len(self._array)
        self._refcount = [1]

    @classmethod
    def from_data(cls, data, subtype):
        return NumpyVectorArray(data)

    @classmethod
    def from_file(cls, path, key=None, single_vector=False, transpose=False):
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
        return NumpyVectorArray(array)

    @classmethod
    def make_array(cls, subtype=None, count=0, reserve=0):
        assert isinstance(subtype, _INDEXTYPES)
        assert count >= 0
        assert reserve >= 0
        va = NumpyVectorArray(np.empty((0, 0)))
        va._array = np.zeros((max(count, reserve), subtype))
        va._len = count
        return va

    @property
    def data(self):
        return self._array[:self._len]

    @property
    def real(self):
        return NumpyVectorArray(self._array[:self._len].real, copy=True)

    @property
    def imag(self):
        return NumpyVectorArray(self._array[:self._len].imag, copy=True)

    @property
    def subtype(self):
        return self._array.shape[1]

    @property
    def dim(self):
        return self._array.shape[1]

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
            ind = set([ind if 0 <= ind else self._len+ind])
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
            C = NumpyVectorArray(self._array)
            C._len = self._len
            C._refcount = self._refcount
            self._refcount[0] += 1
            return C
        else:
            new_array = self._array[_ind]
            if new_array.flags['OWNDATA']:
                new_array = new_array.copy()
            return NumpyVectorArray(new_array)

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

        if NUMPY_INDEX_QUIRK and self._len == 0:
            return

        if isinstance(alpha, np.ndarray):
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

        if NUMPY_INDEX_QUIRK:
            if self._len == 0 and hasattr(_ind, '__len__'):
                _ind = None

        B = x.base._array[x.ind] if x.is_view else x._array[:x._len]
        assert self.len_ind(_ind) == len(B) or len(B) == 1

        alpha_type = type(alpha)
        alpha_dtype = alpha.dtype if alpha_type is np.ndarray else alpha_type
        if self._array.dtype != alpha_dtype or self._array.dtype != B.dtype:
            dtype = np.promote_types(self._array.dtype, alpha_dtype)
            dtype = np.promote_types(dtype, B.dtype)
            self._array = self._array.astype(dtype)

        if np.all(alpha == 0):
            return
        elif np.all(alpha == 1):
            self._array[_ind] += B
        elif np.all(alpha == -1):
            self._array[_ind] -= B
        else:
            if isinstance(alpha, np.ndarray):
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

        return NumpyVectorArray(coefficients.dot(self._array[_ind]))

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
        return 'NumpyVectorArray({})'.format(self._array[:self._len].__str__())

    def __del__(self):
        self._refcount[0] -= 1

    def _deep_copy(self):
        self._array = self._array.copy()  # copy the array data
        self._refcount[0] -= 1            # decrease refcount for original array
        self._refcount = [1]              # create new reference counter


def NumpyVectorSpace(dim):
    """Shorthand for |VectorSpace| `(NumpyVectorArray, dim)`."""
    return VectorSpace(NumpyVectorArray, dim)


class NumpyVectorArrayView(NumpyVectorArray):

    is_view = True

    def __init__(self, array, ind):
        assert array.check_ind(ind)
        self.base = array
        self.ind = array.normalize_ind(ind)

    @property
    def data(self):
        return self.base.data[self.ind]

    @property
    def dim(self):
        return self.base.dim

    @property
    def subtype(self):
        return self.base.subtype

    @property
    def space(self):
        return self.base.space

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
        raise NotImplementedError

    def __del__(self):
        return

    def __repr__(self):
        return 'NumpyVectorArrayView({})'.format(self.data)


_complex_dtypes = (np.complex64, np.complex128)
