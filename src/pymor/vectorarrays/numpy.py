# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Integral, Number

import numpy as np
from scipy.sparse import issparse

from pymor.core.base import classinstancemethod
from pymor.vectorarrays.interface import VectorArray, VectorArrayImpl, VectorSpace, _create_random_values


class NumpyVectorArrayImpl(VectorArrayImpl):

    def __init__(self, array, l=None):
        self._array = array
        self._len = len(array) if l is None else l

    def to_numpy(self, ensure_copy, ind):
        A = self._array[:self._len] if ind is None else self._array[ind]
        if ensure_copy and not A.flags['OWNDATA']:
            return A.copy()
        else:
            return A

    def real(self, ind):
        return NumpyVectorArrayImpl(self.to_numpy(False, ind).real.copy())

    def imag(self, ind):
        return NumpyVectorArrayImpl(self.to_numpy(False, ind).imag.copy())

    def conj(self, ind):
        if np.isrealobj(self._array):
            return self.copy(False, ind)
        return NumpyVectorArrayImpl(np.conj(self.to_numpy(False, ind)))

    def __len__(self):
        return self._len

    def delete(self, ind):
        if ind is None:
            self._array = np.empty((0, self._array.shape[1]))
            self._len = 0
            return
        if type(ind) is slice:
            ind = set(range(*ind.indices(self._len)))
        elif not hasattr(ind, '__len__'):
            ind = {ind if 0 <= ind else self._len + ind}
        else:
            l = self._len
            ind = {i if 0 <= i else l+i for i in ind}
        remaining = sorted(set(range(len(self))) - ind)
        self._array = self._array[remaining]
        self._len = len(self._array)
        if not self._array.flags['OWNDATA']:
            self._array = self._array.copy()

    def copy(self, deep, ind):
        new_array = self._array[:self._len] if ind is None else self._array[ind]
        if not new_array.flags['OWNDATA']:
            new_array = new_array.copy()
        return NumpyVectorArrayImpl(new_array)

    def append(self, other, remove_from_other, oind):
        other_array = other.to_numpy(False, oind)
        len_other = len(other_array)
        if len_other == 0:
            return

        if len_other <= self._array.shape[0] - self._len:
            if self._array.dtype != other_array.dtype:
                self._array = self._array.astype(np.promote_types(self._array.dtype, other_array.dtype), copy=False)
            self._array[self._len:self._len + len_other] = other_array
        else:
            self._array = np.append(self._array[:self._len], other_array, axis=0)
        self._len += len_other

        if remove_from_other:
            other.delete(oind)

    def scal(self, alpha, ind):
        ind = slice(None, self._len) if ind is None else ind
        if type(alpha) is np.ndarray:
            alpha = alpha[:, np.newaxis]

        alpha_type = type(alpha)
        alpha_dtype = alpha.dtype if alpha_type is np.ndarray else alpha_type
        if self._array.dtype != alpha_dtype:
            self._array = self._array.astype(np.promote_types(self._array.dtype, alpha_dtype), copy=False)
        self._array[ind] *= alpha

    def scal_copy(self, alpha, ind):
        ind = slice(None, self._len) if ind is None else ind
        if type(alpha) is np.ndarray:
            alpha = alpha[:, np.newaxis]

        if isinstance(alpha, Number) and alpha == -1:
            return type(self)(- self._array[ind])

        return type(self)(self._array[ind] * alpha)

    def axpy(self, alpha, x, ind, xind):
        ind = slice(None, self._len) if ind is None else ind
        B = x._array[:x._len] if xind is None else x._array[xind]

        alpha_type = type(alpha)
        alpha_dtype = alpha.dtype if alpha_type is np.ndarray else alpha_type
        if self._array.dtype != alpha_dtype or self._array.dtype != B.dtype:
            dtype = np.promote_types(self._array.dtype, alpha_dtype)
            dtype = np.promote_types(dtype, B.dtype)
            self._array = self._array.astype(dtype, copy=False)

        if type(alpha) is np.ndarray:
            if len(B) == 1:
                from scipy.linalg.blas import dger
                a = self._array[ind]
                if len(a) == 0:
                    return
                dger(1., B.ravel(), alpha, a=a.T, overwrite_x=0, overwrite_y=0, overwrite_a=1)
                return
            alpha = alpha[:, np.newaxis]

        if isinstance(alpha, Number):
            if alpha == 1:
                self._array[ind] += B
                return
            elif alpha == -1:
                self._array[ind] -= B
                return

        self._array[ind] += B * alpha

    def axpy_copy(self, alpha, x, ind, xind):
        ind = slice(None, self._len) if ind is None else ind
        B = x._array[:x._len] if xind is None else x._array[xind]

        if type(alpha) is np.ndarray:
            alpha = alpha[:, np.newaxis]

        if isinstance(alpha, Number):
            if alpha == 1:
                return type(self)(self._array[ind] + B)
            elif alpha == -1:
                return type(self)(self._array[ind] - B)

        return type(self)(self._array[ind] + B * alpha)

    def inner(self, other, ind, oind):
        A = self._array[:self._len] if ind is None else self._array[ind]
        B = other._array[:other._len] if oind is None else other._array[oind]

        # .conj() is a no-op on non-complex data types
        return A.conj().dot(B.T)

    def pairwise_inner(self, other, ind, oind):
        A = self._array[:self._len] if ind is None else self._array[ind]
        B = other._array[:other._len] if oind is None else other._array[oind]

        # .conj() is a no-op on non-complex data types
        return np.sum(A.conj() * B, axis=1)

    def lincomb(self, coefficients, ind):
        A = self._array[:self._len] if ind is None else self._array[ind]
        return NumpyVectorArrayImpl(coefficients.dot(A))

    def norm(self, ind):
        A = self._array[:self._len] if ind is None else self._array[ind]
        return np.linalg.norm(A, axis=1)

    def norm2(self, ind):
        A = self._array[:self._len] if ind is None else self._array[ind]
        return np.sum((A * A.conj()).real, axis=1)

    def dofs(self, dof_indices, ind):
        ind = slice(None, self._len) if ind is None else ind
        return self._array[:, dof_indices][ind, :]

    def amax(self, ind):
        ind = slice(None, self._len) if ind is None else ind
        A = np.abs(self._array[ind])
        max_ind = np.argmax(A, axis=1)
        max_val = A[np.arange(len(A)), max_ind]
        return max_ind, max_val


class NumpyVectorArray(VectorArray):
    """|VectorArray| implementation via |NumPy arrays|.

    This is the default |VectorArray| type used by all |Operators|
    in pyMOR's discretization toolkit. Moreover, all reduced |Operators|
    are based on |NumpyVectorArray|.

    This class is just a thin wrapper around the underlying
    |NumPy array|. Thus, while operations like
    :meth:`~pymor.vectorarrays.interface.VectorArray.axpy` or
    :meth:`~pymor.vectorarrays.interface.VectorArray.inner`
    will be quite efficient, removing or appending vectors will
    be costly.

    .. warning::
        This class is not intended to be instantiated directly. Use
        the associated :class:`VectorSpace <NumpyVectorSpace>` instead.
    """

    impl_type = NumpyVectorArrayImpl

    def __str__(self):
        return str(self.to_numpy())

    def _format_repr(self, max_width, verbosity):
        return super()._format_repr(max_width, verbosity, override={'impl': str(self.to_numpy())})


class NumpyVectorSpace(VectorSpace):
    """|VectorSpace| of |NumpyVectorArrays|.

    Parameters
    ----------
    dim
        The dimension of the vectors contained in the space.
    """

    def __init__(self, dim):
        assert isinstance(dim, Integral)
        self.dim = int(dim)

    def __eq__(self, other):
        return type(other) is type(self) and self.dim == other.dim

    def __hash__(self):
        return hash(self.dim)

    def zeros(self, count=1, reserve=0):
        assert count >= 0
        assert reserve >= 0
        return NumpyVectorArray(self, NumpyVectorArrayImpl(np.zeros((max(count, reserve), self.dim)), count))

    def full(self, value, count=1, reserve=0):
        assert count >= 0
        assert reserve >= 0
        return NumpyVectorArray(self, NumpyVectorArrayImpl(np.full((max(count, reserve), self.dim), value), count))

    def random(self, count=1, distribution='uniform', reserve=0, **kwargs):
        assert count >= 0
        assert reserve >= 0
        va = self.zeros(count, reserve)
        va.impl._array[:count] = _create_random_values((count, self.dim), distribution, **kwargs)
        return va

    @classinstancemethod
    def make_array(cls, obj):  # noqa: N805
        return cls._array_factory(obj)

    @make_array.instancemethod
    def make_array(self, obj):
        """:noindex:"""  # noqa: D400
        return self._array_factory(obj, space=self)

    @classinstancemethod
    def from_numpy(cls, data, ensure_copy=False):  # noqa: N805
        return cls._array_factory(data.copy() if ensure_copy else data)

    @from_numpy.instancemethod
    def from_numpy(self, data, ensure_copy=False):
        """:noindex:"""  # noqa: D400
        return self._array_factory(data.copy() if ensure_copy else data, space=self)

    @classinstancemethod
    def from_file(cls, path, key=None, single_vector=False, transpose=False):  # noqa: N805
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
        return cls.make_array(array)

    @from_file.instancemethod
    def from_file(self, path, key=None, single_vector=False, transpose=False):
        """:noindex:"""  # noqa: D400
        return type(self).from_file(path, key=key, single_vector=single_vector, transpose=transpose)

    @classmethod
    def _array_factory(cls, array, space=None):
        if type(array) is np.ndarray:
            pass
        elif issparse(array):
            array = array.toarray()
        else:
            array = np.array(array, ndmin=2)
        if array.ndim != 2:
            assert array.ndim == 1
            array = np.reshape(array, (1, -1))
        if space is None:
            return NumpyVectorArray(cls(array.shape[1]), NumpyVectorArrayImpl(array))
        else:
            assert array.shape[1] == space.dim
            return NumpyVectorArray(space, NumpyVectorArrayImpl(array))

    @property
    def is_scalar(self):
        return self.dim == 1
