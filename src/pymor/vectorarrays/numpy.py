# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Integral, Number
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from typing_extensions import Self

from pymor.core.base import classinstancemethod
from pymor.vectorarrays.interface import Ind, VectorArray, VectorArrayImpl, VectorSpace, _create_random_values


class NumpyVectorArrayImpl(VectorArrayImpl):

    def __init__(self, array: np.ndarray, l: int | None = None):
        self._array = array
        self._len = array.shape[1] if l is None else l

    def to_numpy(self, ensure_copy: bool, ind: Ind) -> np.ndarray:
        A = self._array[:, :self._len] if ind is None else self._array[:, ind]
        if ensure_copy and not A.flags['OWNDATA']:
            return A.copy(order='F')
        else:
            return A

    def real(self, ind: Ind) -> Self:
        return type(self)(self.to_numpy(False, ind).real.copy(order='F'))

    def imag(self, ind: Ind) -> Self:
        return type(self)(self.to_numpy(False, ind).imag.copy(order='F'))

    def conj(self, ind: Ind) -> Self:
        if np.isrealobj(self._array):
            return self.copy(False, ind)
        return type(self)(np.conj(self.to_numpy(False, ind), order='F'))

    def __len__(self) -> int:
        return self._len

    def delete(self, ind: Ind) -> None:
        if ind is None:
            self._array = np.empty((self._array.shape[0], 0))
            self._len = 0
            return
        if isinstance(ind, slice):
            ind_set = set(range(*ind.indices(self._len)))
        elif isinstance(ind, int):
            ind_set = {ind if 0 <= ind else self._len + ind}
        else:
            l = self._len
            ind_set = {i if 0 <= i else l+i for i in ind}
        remaining = sorted(set(range(len(self))) - ind_set)
        self._array = self._array[:, remaining]
        self._len = self._array.shape[1]
        if not self._array.flags['OWNDATA']:
            self._array = self._array.copy(order='F')

    def copy(self, deep: bool, ind: Ind) -> Self:
        del deep # unused
        new_array = self._array[:, :self._len] if ind is None else self._array[:, ind]
        if not new_array.flags['OWNDATA']:
            new_array = new_array.copy(order='F')
        return type(self)(new_array)

    def append(self, other: Self, remove_from_other: bool, oind: Ind) -> None:
        other_array = other.to_numpy(False, oind)
        len_other = other_array.shape[1]
        if len_other == 0:
            return

        if len_other <= self._array.shape[1] - self._len:
            if self._array.dtype != other_array.dtype:
                self._array = self._array.astype(np.promote_types(self._array.dtype, other_array.dtype), copy=False)
            self._array[:, self._len:self._len + len_other] = other_array
        else:
            new_array = np.empty((self._array.shape[0], self._len + len_other),
                                 dtype=np.promote_types(self._array.dtype, other_array.dtype), order='F')
            np.concatenate((self._array[:, :self._len], other_array), axis=1, out=new_array)
            self._array = new_array
        self._len += len_other

        if remove_from_other:
            other.delete(oind)

    def scal(self, alpha: float | complex, ind: Ind) -> None:
        ind = slice(None, self._len) if ind is None else ind
        if type(alpha) is np.ndarray:
            alpha = alpha[np.newaxis, :]

        alpha_dtype = alpha.dtype if isinstance(alpha, np.ndarray) else type(alpha)
        if self._array.dtype != alpha_dtype:
            new_dtype = np.promote_types(self._array.dtype, alpha_dtype)
            if new_dtype != self._array.dtype:
                self._array = self._array.astype(new_dtype, order='F')

        if ind is not None and not self._array.flags.f_contiguous:
            self._array = self._array.copy(order='F')

        self._array[:, ind] *= alpha

    def scal_copy(self, alpha: float | complex, ind: Ind) -> Self:
        ind = slice(None, self._len) if ind is None else ind
        if type(alpha) is np.ndarray:
            alpha = alpha[np.newaxis, :]

        if isinstance(alpha, Number) and alpha == -1:
            return type(self)(- self._array[:, ind])

        return type(self)(np.multiply(self._array[:, ind], alpha, order='F'))

    def axpy(self, alpha: float | complex, x: Self, ind: Ind, xind: Ind) -> None:
        ind = slice(None, self._len) if ind is None else ind
        B = x._array[:, :x._len] if xind is None else x._array[:, xind]

        alpha_dtype = alpha.dtype if isinstance(alpha, np.ndarray) else type(alpha)
        if self._array.dtype != alpha_dtype or self._array.dtype != B.dtype:
            new_dtype = np.promote_types(self._array.dtype, alpha_dtype)
            new_dtype = np.promote_types(new_dtype, B.dtype)
            if new_dtype != self._array.dtype:
                self._array = self._array.astype(new_dtype, order='F')

        if type(alpha) is np.ndarray:
            alpha = alpha[np.newaxis, :]

        if isinstance(alpha, Number):
            if alpha == 1:
                self._array[:, ind] += B
                return
            elif alpha == -1:
                self._array[:, ind] -= B
                return

        if ind is not None and not self._array.flags.f_contiguous:
            self._array = self._array.copy(order='F')

        self._array[:, ind] += B * alpha

    def axpy_copy(self, alpha: float | complex, x: Self, ind: Ind, xind: Ind) -> Self:
        ind = slice(None, self._len) if ind is None else ind
        B = x._array[:, :x._len] if xind is None else x._array[:, xind]

        if type(alpha) is np.ndarray:
            alpha = alpha[np.newaxis, :]

        if isinstance(alpha, Number):
            if alpha == 1:
                return type(self)(np.add(self._array[:, ind],  B, order='F'))
            elif alpha == -1:
                return type(self)(np.subtract(self._array[:, ind],  B, order='F'))

        return type(self)(np.add(self._array[:, ind], B * alpha, order='F'))

    def inner(self, other: Self, ind: Ind, oind: Ind) -> np.ndarray:
        if ind is not None and not self._array.flags.f_contiguous:
            self._array = self._array.copy(order='F')

        A = self._array[:, :self._len] if ind is None else self._array[:, ind]
        B = other._array[:, :other._len] if oind is None else other._array[:, oind]

        # .conj() is a no-op on non-complex data types
        return np.matmul(A.conj().T, B, order='F')

    def pairwise_inner(self, other: Self, ind: Ind, oind: Ind) -> np.ndarray:
        if ind is not None and not self._array.flags.f_contiguous:
            self._array = self._array.copy(order='F')

        A = self._array[:, :self._len] if ind is None else self._array[:, ind]
        B = other._array[:, :other._len] if oind is None else other._array[:, oind]

        # .conj() is a no-op on non-complex data types
        return np.sum(A.conj() * B, axis=0)

    def lincomb(self, coefficients: np.ndarray, ind: Ind) -> Self:
        if ind is not None and not self._array.flags.f_contiguous:
            self._array = self._array.copy(order='F')

        A = self._array[:, :self._len] if ind is None else self._array[:, ind]
        return type(self)(np.matmul(A, coefficients, order='F'))

    def norm(self, ind: Ind) -> np.ndarray:
        if ind is not None and not self._array.flags.f_contiguous:
            self._array = self._array.copy(order='F')

        A = self._array[:, :self._len] if ind is None else self._array[:, ind]
        return np.linalg.norm(A, axis=0)

    def norm2(self, ind: Ind) -> np.ndarray:
        if ind is not None and not self._array.flags.f_contiguous:
            self._array = self._array.copy(order='F')

        A = self._array[:, :self._len] if ind is None else self._array[:, ind]
        return np.sum((A * A.conj()).real, axis=0)

    def dofs(self, dof_indices: np.ndarray, ind: Ind) -> np.ndarray:
        if ind is not None and not self._array.flags.f_contiguous:
            self._array = self._array.copy(order='F')

        ind = slice(None, self._len) if ind is None else ind
        return np.asfortranarray(self._array[dof_indices, :][:, ind])

    def amax(self, ind: Ind) -> tuple[np.ndarray, np.ndarray]:
        if ind is not None and not self._array.flags.f_contiguous:
            self._array = self._array.copy(order='F')

        ind = slice(None, self._len) if ind is None else ind
        A = np.abs(self._array[:, ind])
        max_ind = np.argmax(A, axis=0)
        max_val = A[max_ind, np.arange(A.shape[1])]
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

    def __str__(self) -> str:
        return str(self.to_numpy())

    def _format_repr(self, max_width, verbosity, override={}):
        del override # ignored
        return super()._format_repr(max_width, verbosity, override={'impl': str(self.to_numpy())})


class NumpyVectorSpace(VectorSpace):
    """|VectorSpace| of |NumpyVectorArrays|.

    Parameters
    ----------
    dim
        The dimension of the vectors contained in the space.
    """

    def __init__(self, dim: int):
        assert isinstance(dim, Integral)
        self.dim = dim = int(dim)
        self.is_scalar = (dim == 1)

    def __eq__(self, other) -> bool:
        return type(other) is type(self) and self.dim == other.dim

    def __hash__(self):
        return hash(self.dim)

    def zeros(self, count: int = 1, reserve: int = 0) -> NumpyVectorArray:
        assert count >= 0
        assert reserve >= 0
        return NumpyVectorArray(self, NumpyVectorArrayImpl(np.zeros((self.dim, max(count, reserve)), order='F'), count))

    def full(self, value, count: int = 1, reserve: int = 0) -> NumpyVectorArray:
        assert count >= 0
        assert reserve >= 0
        return NumpyVectorArray(self,
                                NumpyVectorArrayImpl(np.full((self.dim, max(count, reserve)), value, order='F'), count))

    def random(self, count: int = 1, distribution: str = 'uniform', reserve: int = 0, **kwargs) -> NumpyVectorArray:
        assert count >= 0
        assert reserve >= 0
        va = self.zeros(count, reserve)
        assert isinstance(va.impl, NumpyVectorArrayImpl)
        va.impl._array[:, :count] = _create_random_values((self.dim, count), distribution, **kwargs)
        return va

    @classinstancemethod
    def make_array(cls, obj) -> NumpyVectorArray:  # type: ignore # noqa: N805
        return cls._array_factory(obj)

    @make_array.instancemethod
    def _(self, obj) -> NumpyVectorArray:
        """:noindex:"""  # noqa: D400
        return self._array_factory(obj, space=self)

    @classinstancemethod
    def from_numpy(cls, data: np.ndarray, ensure_copy: bool = False):  # type: ignore # noqa: N805
        return cls._array_factory(data.copy(order='F') if ensure_copy else data)

    @from_numpy.instancemethod
    def _(self, data: np.ndarray, ensure_copy: bool = False):
        """:noindex:"""  # noqa: D400
        return self._array_factory(data.copy(order='F') if ensure_copy else data, space=self)

    @classinstancemethod
    def from_file(cls, path: str | Path, key: str | None = None, single_vector: bool = False, # noqa: N805
                  transpose: bool = False):
        assert not (single_vector and transpose)
        from pymor.tools.io import load_matrix
        array = load_matrix(path, key=key)
        assert isinstance(array, np.ndarray)
        assert array.ndim <= 2
        if array.ndim == 1:
            array = array.reshape((-1, 1))
        if single_vector:
            assert array.shape[0] == 1 or array.shape[1] == 1
            array = array.reshape((-1, 1))
        if transpose:
            array = array.T
        return cls.make_array(np.asfortranarray(array))

    def _(self, path: str | Path, key: str | None = None, single_vector: bool = False,
          transpose: bool = False):
        """:noindex:"""  # noqa: D400
        return type(self).from_file(path, key=key, single_vector=single_vector, transpose=transpose)

    @classmethod
    def _array_factory(cls, array, space=None) -> NumpyVectorArray:
        if type(array) is np.ndarray:
            pass
        elif issparse(array):
            array = array.toarray() # type: ignore
        else:
            array = np.array(array, ndmin=2)
        if array.ndim != 2:
            assert array.ndim == 1
            array = np.reshape(array, (-1, 1))
        if space is None:
            return NumpyVectorArray(cls(array.shape[0]), NumpyVectorArrayImpl(array))
        else:
            assert array.shape[0] == space.dim
            return NumpyVectorArray(space, NumpyVectorArrayImpl(array))
