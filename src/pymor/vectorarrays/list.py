# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typing import Self

import numpy as np

from pymor.core.base import BasicObject, abstractclassmethod, abstractmethod, classinstancemethod
from pymor.core.typing import Complex, Real
from pymor.vectorarrays.interface import Ind, VectorArray, VectorArrayImpl, VectorSpace, _create_random_values


class Vector(BasicObject):
    """Interface for vectors used in conjunction with |ListVectorArray|.

    This interface must be satisfied by the individual entries of the
    vector `list` managed by |ListVectorArray|. All interface methods
    have a direct counterpart in the |VectorArray| interface.
    """

    @abstractmethod
    def copy(self, deep: bool = False) -> Self: ...

    @abstractmethod
    def scal(self, alpha: Complex) -> None: ...

    @abstractmethod
    def axpy(self, alpha: Complex, x: Self) -> None: ...

    def inner(self, other: Self) -> Complex:
        raise NotImplementedError

    @abstractmethod
    def norm(self) -> float: ...

    @abstractmethod
    def norm2(self) -> float: ...

    def sup_norm(self) -> float:
        _, max_val = self.amax()
        return max_val

    @abstractmethod
    def dofs(self, dof_indices: list[int] | np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def amax(self) -> tuple[int, float]: ...

    def to_numpy(self, ensure_copy: bool = False) -> np.ndarray:
        raise NotImplementedError

    def __add__(self, other: Self) -> Self:
        result = self.copy()
        result.axpy(1, other)
        return result

    def __iadd__(self, other: Self) -> Self:
        self.axpy(1, other)
        return self

    __radd__ = __add__

    def __sub__(self, other: Self) -> Self:
        result = self.copy()
        result.axpy(-1, other)
        return result

    def __isub__(self, other: Self) -> Self:
        self.axpy(-1, other)
        return self

    def __mul__(self, other: Complex) -> Self:
        result = self.copy()
        result.scal(other)
        return result

    def __imul__(self, other: Complex) -> Self:
        self.scal(other)
        return self

    def __neg__(self) -> Self:
        result = self.copy()
        result.scal(-1)
        return result

    @property
    def real(self) -> Self:
        return self.copy()

    @property
    def imag(self) -> Self | None:
        return None

    def conj(self) -> Self:
        return self.copy()


class CopyOnWriteVector(Vector):

    @abstractclassmethod
    def from_instance(cls, instance: Self) -> Self: ...  # noqa: N805

    @abstractmethod
    def _copy_data(self) -> None: ...

    @abstractmethod
    def _scal(self, alpha: Complex) -> None: ...

    @abstractmethod
    def _axpy(self, alpha: Complex, x: Self) -> None: ...

    def copy(self, deep: bool = False) -> Self:
        c = self.from_instance(self)
        if deep:
            c._copy_data()
        else:
            try:
                self._refcount[0] += 1
            except AttributeError:
                self._refcount = [2]
            c._refcount = self._refcount
        return c

    def scal(self, alpha: Complex) -> None:
        self._copy_data_if_needed()
        self._scal(alpha)

    def axpy(self, alpha: Complex, x: Self) -> None:
        self._copy_data_if_needed()
        self._axpy(alpha, x)

    def __del__(self) -> None:
        try:
            self._refcount[0] -= 1
        except AttributeError:
            pass

    def _copy_data_if_needed(self) -> None:
        try:
            if self._refcount[0] > 1:
                self._refcount[0] -= 1
                self._copy_data()
                self._refcount = [1]
        except AttributeError:
            self._refcount = [1]


class ComplexifiedVector(Vector):

    def __init__(self, real_part: Vector, imag_part: Vector | None):
        self.real_part, self.imag_part = real_part, imag_part

    def copy(self, deep: bool = False) -> Self:
        real_part = self.real_part.copy(deep=deep)
        imag_part = self.imag_part.copy(deep=deep) if self.imag_part is not None else None
        return type(self)(real_part, imag_part)

    def scal(self, alpha: Complex) -> None:
        if self.imag_part is None:
            if alpha.imag != 0:
                self.imag_part = self.real_part * alpha.imag
            self.real_part.scal(alpha.real)
        else:
            if alpha.imag == 0:
                self.real_part.scal(alpha.real)
                self.imag_part.scal(alpha.real)
            else:
                old_real_part = self.real_part.copy()
                self.real_part.scal(alpha.real)
                self.real_part.axpy(-alpha.imag, self.imag_part)
                self.imag_part.scal(alpha.real)
                self.imag_part.axpy(alpha.imag, old_real_part)

    def axpy(self, alpha: Complex, x: Self) -> None:
        if x is self:
            self.scal(1. + alpha)
            return

        # real part
        self.real_part.axpy(alpha.real, x.real_part)
        if x.imag_part is not None:
            self.real_part.axpy(-alpha.imag, x.imag_part)

        # imaginary part
        if alpha.imag != 0:
            if self.imag_part is None:
                self.imag_part = x.real_part * alpha.imag
            else:
                self.imag_part.axpy(alpha.imag, x.real_part)
        if x.imag_part is not None:
            if self.imag_part is None:
                self.imag_part = x.imag_part * alpha.real
            else:
                self.imag_part.axpy(alpha.real, x.imag_part)

    def inner(self, other: Self) -> Complex:
        result = self.real_part.inner(other.real_part)
        if self.imag_part is not None:
            result += self.imag_part.inner(other.real_part) * (-1j)
        if other.imag_part is not None:
            result += self.real_part.inner(other.imag_part) * 1j
        if self.imag_part is not None and other.imag_part is not None:
            result += self.imag_part.inner(other.imag_part)
        return result

    def norm(self) -> float:
        result = self.real_part.norm()
        if self.imag_part is not None:
            result = np.linalg.norm([result, self.imag_part.norm()])
        assert isinstance(result, float)
        return result

    def norm2(self) -> float:
        result = self.real_part.norm2()
        if self.imag_part is not None:
            result += self.imag_part.norm2()
        return result

    def sup_norm(self) -> float:
        if self.imag_part is not None:
            # we cannot compute the sup_norm from the sup_norms of real_part and imag_part
            return self.amax()[1]
        return self.real_part.sup_norm()

    def dofs(self, dof_indices: list[int] | np.ndarray) -> np.ndarray:
        values = self.real_part.dofs(dof_indices)
        if self.imag_part is not None:
            imag_values = self.imag_part.dofs(dof_indices)
            return values + imag_values * 1j
        else:
            return values

    def amax(self) -> tuple[int, float]:
        if self.imag_part is None:
            return self.real_part.amax()
        else:
            A = np.abs(self.real_part.to_numpy() + self.imag_part.to_numpy() * 1j)
            max_ind = np.argmax(A).item()
            return max_ind, A[max_ind]

    def to_numpy(self, ensure_copy: bool = False) -> np.ndarray:
        if self.imag_part is not None:
            return self.real_part.to_numpy(ensure_copy=False) + self.imag_part.to_numpy(ensure_copy=False) * 1j
        else:
            return self.real_part.to_numpy(ensure_copy=ensure_copy)

    @property
    def real(self) -> Self:
        return type(self)(self.real_part.copy(), None)

    @property
    def imag(self) -> Self | None:
        return type(self)(self.imag_part.copy(), None) if self.imag_part is not None else None

    def conj(self) -> Self:
        return type(self)(self.real_part.copy(), -self.imag_part if self.imag_part is not None else None)


class NumpyVector(CopyOnWriteVector):
    """Vector stored in a NumPy 1D-array."""

    def __init__(self, array: np.ndarray):
        self._array = array

    @classmethod
    def from_instance(cls, instance: Self) -> Self:
        return cls(instance._array)

    def to_numpy(self, ensure_copy: bool = False) -> np.ndarray:
        if ensure_copy:
            return self._array.copy()
        else:
            self._copy_data_if_needed()
            return self._array

    @property
    def dim(self) -> int:
        return len(self._array)

    def _copy_data(self) -> None:
        self._array = self._array.copy()

    def _scal(self, alpha: Complex) -> None:
        try:
            self._array *= alpha
        except TypeError:  # e.g. when scaling real array by complex alpha
            self._array = self._array * alpha

    def _axpy(self, alpha: Complex, x: Self) -> None:
        assert self.dim == x.dim
        if alpha == 0:
            return
        if alpha == 1:
            try:
                self._array += x._array
            except TypeError:
                self._array = self._array + x._array
        elif alpha == -1:
            try:
                self._array -= x._array
            except TypeError:
                self._array = self._array - x._array
        else:
            try:
                self._array += x._array * alpha
            except TypeError:
                self._array = self._array + x._array * alpha

    def inner(self, other: Self) -> Complex:
        assert self.dim == other.dim
        return np.sum(self._array.conj() * other._array)

    def norm(self) -> float:
        return np.linalg.norm(self._array).item()

    def norm2(self):
        return np.sum((self._array * self._array.conj()).real)

    def dofs(self, dof_indices):
        return self._array[dof_indices]

    def amax(self) -> tuple[int, float]:
        A = np.abs(self._array)
        max_ind = np.argmax(A).item()
        max_val = A[max_ind]
        return max_ind, np.abs(max_val)

    @property
    def real(self) -> Self:
        return self.__class__(self._array.real.copy())

    @property
    def imag(self) -> Self:
        return self.__class__(self._array.imag.copy())

    def conj(self) -> Self:
        return self.__class__(self._array.conj())


class ListVectorArrayImpl(VectorArrayImpl):

    _NONE = ()

    def __init__(self, vectors: list[Vector], space: 'ListVectorSpace'):
        self._list = vectors
        self.space = space

    def _indexed(self, ind: Ind) -> list[Vector]:
        if ind is None:
            return self._list
        elif isinstance(ind, slice):
            return self._list[ind]
        elif isinstance(ind, list):
            return [self._list[i] for i in ind]
        else:
            assert False

    def to_numpy(self, ensure_copy: bool, ind: Ind) -> np.ndarray:
        del ensure_copy # unused
        vectors = [v.to_numpy() for v in self._indexed(ind)]
        if vectors:
            return np.array(vectors).T
        else:
            return np.empty((self.space.dim, 0))

    def __len__(self) -> int:
        return len(self._list)

    def delete(self, ind: Ind) -> None:
        if ind is None:
            del self._list[:]
        elif isinstance(ind, list):
            thelist = self._list
            l = len(thelist)
            remaining = sorted(set(range(l)) - {i if 0 <= i else l+i for i in ind})
            self._list = [thelist[i] for i in remaining]
        else:
            del self._list[ind]

    def append(self, other: Self, remove_from_other: bool, oind: Ind) -> None:
        if not remove_from_other:
            self._list.extend([v.copy() for v in other._indexed(oind)])
        else:
            self._list.extend(other._indexed(oind))
            other.delete(oind)

    def copy(self, deep: bool, ind: Ind) -> Self:
        return type(self)([v.copy(deep=deep) for v in self._indexed(ind)], self.space)

    def scal(self, alpha: Complex | np.ndarray, ind: Ind) -> None:
        if isinstance(alpha, np.ndarray):
            for a, v in zip(alpha, self._indexed(ind), strict=True):
                v.scal(a)
        else:
            for v in self._indexed(ind):
                v.scal(alpha)

    def axpy(self, alpha: Complex | np.ndarray, x: Self, ind: Ind, xind: Ind) -> None:
        if np.all(alpha == 0):
            return

        if self is x:
            x_list = x.copy(False, xind)._list
        else:
            x_list = x._indexed(xind)

        if x.len_ind(xind) == 1:
            xx = next(iter(x_list))
            if isinstance(alpha, np.ndarray):
                for a, y in zip(alpha, self._indexed(ind), strict=True):
                    y.axpy(a, xx)
            else:
                for y in self._indexed(ind):
                    y.axpy(alpha, xx)
        else:
            if isinstance(alpha, np.ndarray):
                for a, xx, y in zip(alpha, x_list, self._indexed(ind), strict=True):
                    y.axpy(a, xx)
            else:
                for xx, y in zip(x_list, self._indexed(ind), strict=True):
                    y.axpy(alpha, xx)

    def inner(self, other: Self, ind: Ind, oind: Ind) -> np.ndarray:
        return (np.array([[a.inner(b) for b in other._indexed(oind)] for a in self._indexed(ind)])
                  .reshape((self.len_ind(ind), other.len_ind(oind))))

    def pairwise_inner(self, other: Self, ind: Ind, oind: Ind) -> np.ndarray:
        return np.array([a.inner(b) for a, b in zip(self._indexed(ind), other._indexed(oind), strict=True)])

    def gramian(self, ind: Ind) -> np.ndarray:
        self_list = self._indexed(ind)
        l = len(self_list)
        if l == 0:
            return np.zeros((0,0))
        R: list[list[Complex]] = [[0.] * l for _ in range(l)]
        for i in range(l):
            for j in range(i, l):
                R[i][j] = self_list[i].inner(self_list[j])
                if i == j:
                    R[i][j] = R[i][j].real
                else:
                    R[j][i] = R[i][j].conjugate()
        return np.array(R)

    def lincomb(self, coefficients: np.ndarray, ind: Ind) -> Self:
        RL = []
        for coeffs in coefficients.T:
            R = self.space.zero_vector()
            for v, c in zip(self._indexed(ind), coeffs, strict=True):
                R.axpy(c, v)
            RL.append(R)

        return type(self)(RL, self.space)

    def norm(self, ind: Ind) -> np.ndarray:
        return np.array([v.norm() for v in self._indexed(ind)])

    def norm2(self, ind: Ind) -> np.ndarray:
        return np.array([v.norm2() for v in self._indexed(ind)])

    def dofs(self, dof_indices: np.ndarray, ind: Ind) -> np.ndarray:
        return (np.array([v.dofs(dof_indices) for v in self._indexed(ind)])
                  .reshape((self.len_ind(ind), len(dof_indices)))).T

    def amax(self, ind: Ind) -> tuple[np.ndarray, np.ndarray]:
        l = self.len_ind(ind)
        MI = np.empty(l, dtype=int)
        MV = np.empty(l)

        for k, v in enumerate(self._indexed(ind)):
            MI[k], MV[k] = v.amax()

        return MI, MV

    def real(self, ind: Ind) -> Self:
        return type(self)([v.real for v in self._indexed(ind)], self.space)

    def imag(self, ind: Ind) -> Self:
        # note that Vector.imag is allowed to return None in case
        # of a real vector, so we have to check for that.
        # returning None is allowed as ComplexifiedVector does not know
        # how to create a new zero vector.
        return type(self)([v.imag or self.space.zero_vector() for v in self._indexed(ind)], self.space)

    def conj(self, ind: Ind) -> Self:
        return type(self)([v.conj() for v in self._indexed(ind)], self.space)


class ListVectorArray(VectorArray):
    """|VectorArray| implemented as a Python list of vectors.

    This |VectorArray| implementation is the first choice when
    creating pyMOR wrappers for external solvers which are based
    on single vector objects. In order to do so, a wrapping
    subclass of :class:`Vector` has to be provided
    on which the implementation of |ListVectorArray| will operate.
    The associated |VectorSpace| is a subclass of
    :class:`ListVectorSpace`.

    For an example, see :class:`NumpyVector` and :class:`NumpyListVectorSpace`,
    :class:`~pymor.bindings.fenics.FenicsVector` and
    :class:`~pymor.bindings.fenics.FenicsVectorSpace`,
    :class:`~pymor.bindings.ngsolve.NGSolveVector` and
    :class:`~pymor.bindings.ngsolve.NGSolveVectorSpace`.
    """

    impl_type = ListVectorArrayImpl

    def __str__(self) -> str:
        return f'{type(self).__name__} of {len(self.impl._list)} vectors of space {self.space}'

    @property
    def vectors(self) -> list[Vector]:
        return self.impl._indexed(self.ind)


class ListVectorSpace[V: Vector](VectorSpace):
    """|VectorSpace| of |ListVectorArrays|."""

    vector_type: type[V]
    array_type = ListVectorArray

    @abstractmethod
    def zero_vector(self) -> V: ...

    def ones_vector(self) -> V:
        return self.full_vector(1.)

    def full_vector(self, value: Complex) -> V:
        return self.vector_from_numpy(np.full(self.dim, value))

    def random_vector(self, distribution: 'str', **kwargs) -> V:
        values = _create_random_values(self.dim, distribution, **kwargs)
        return self.vector_from_numpy(values)

    @abstractmethod
    def make_vector(self, obj) -> V: ...

    def vector_from_numpy(self, data: np.ndarray, ensure_copy=False) -> V:
        raise NotImplementedError

    @classmethod
    def space_from_vector_obj(cls, vec: V) -> Self:
        raise NotImplementedError

    @classmethod
    def space_from_dim(cls, dim: int) -> Self:
        raise NotImplementedError

    def zeros(self, count: int = 1, reserve: int = 0) -> ListVectorArray:
        assert count >= 0
        assert reserve >= 0
        return ListVectorArray(self, ListVectorArrayImpl([self.zero_vector() for _ in range(count)], self))

    def ones(self, count: int = 1, reserve: int = 0) -> ListVectorArray:
        assert count >= 0
        assert reserve >= 0
        return ListVectorArray(self, ListVectorArrayImpl([self.ones_vector() for _ in range(count)], self))

    def full(self, value: Complex, count: int = 1, reserve: int = 0) -> ListVectorArray:
        assert count >= 0
        assert reserve >= 0
        return ListVectorArray(self, ListVectorArrayImpl([self.full_vector(value) for _ in range(count)], self))

    def random(self, count: int = 1, distribution: str = 'uniform', reserve: int = 0, **kwargs) -> ListVectorArray:
        assert count >= 0
        assert reserve >= 0
        return ListVectorArray(
            self,
            ListVectorArrayImpl([self.random_vector(distribution=distribution, **kwargs)
                                 for _ in range(count)], self)
        )

    @classinstancemethod
    def make_array(cls, obj) -> ListVectorArray:  # type: ignore # noqa: N805
        if len(obj) == 0:
            raise NotImplementedError
        return cls.space_from_vector_obj(obj[0]).make_array(obj)

    @make_array.instancemethod
    def _(self, obj) -> ListVectorArray:
        """:noindex:"""  # noqa: D400
        return ListVectorArray(
            self,
            ListVectorArrayImpl([v if isinstance(v, self.vector_type) else self.make_vector(v) for v in obj], self)
        )

    @classinstancemethod
    def from_numpy(cls, data: np.ndarray, ensure_copy: bool = False):  # type: ignore # noqa: N805
        return cls.space_from_dim(data.shape[0]).from_numpy(data, ensure_copy=ensure_copy)

    @from_numpy.instancemethod
    def _(self, data: np.ndarray, ensure_copy: bool = False) -> ListVectorArray:
        """:noindex:"""  # noqa: D400
        if data.ndim == 1:
            data = data.reshape((-1, 1))
        return ListVectorArray(
            self, ListVectorArrayImpl([self.vector_from_numpy(v, ensure_copy=ensure_copy) for v in data.T], self)
        )


class ComplexifiedListVectorSpace[R: Vector](ListVectorSpace):

    real_vector_type: type[R]
    vector_type = ComplexifiedVector

    @abstractmethod
    def real_zero_vector(self) -> R: ...

    def zero_vector(self) -> ComplexifiedVector:
        return self.vector_type(self.real_zero_vector(), None)

    def real_full_vector(self, value: Real) -> R:
        return self.real_vector_from_numpy(np.full(self.dim, value))

    def full_vector(self, value: Complex) -> ComplexifiedVector:
        real_part = self.real_full_vector(value.real)
        imag_part = self.real_full_vector(value.imag) if value.imag != 0. else None
        return self.vector_type(real_part, imag_part)

    def real_random_vector(self, distribution, **kwargs) -> R:
        values = _create_random_values(self.dim, distribution, **kwargs)
        return self.real_vector_from_numpy(values)

    def random_vector(self, distribution, **kwargs) -> ComplexifiedVector:
        return self.vector_type(self.real_random_vector(distribution, **kwargs), None)

    @abstractmethod
    def real_make_vector(self, obj) -> R: ...

    def make_vector(self, obj) -> ComplexifiedVector:
        if isinstance(obj, self.real_vector_type):
            return self.vector_type(obj, None)
        else:
            return self.vector_type(self.real_make_vector(obj), None)

    def real_vector_from_numpy(self, data, ensure_copy=False) -> R:
        raise NotImplementedError

    def vector_from_numpy(self, data: np.ndarray, ensure_copy=False) -> ComplexifiedVector:
        if np.iscomplexobj(data):
            real_part = self.real_vector_from_numpy(data.real)
            imag_part = self.real_vector_from_numpy(data.imag)
        else:
            real_part = self.real_vector_from_numpy(data, ensure_copy=ensure_copy)
            imag_part = None
        return self.vector_type(real_part, imag_part)


class NumpyListVectorSpace(ListVectorSpace):

    vector_type = NumpyVector

    def __init__(self, dim):
        self.dim = dim

    def __eq__(self, other) -> bool:
        return type(other) is NumpyListVectorSpace and self.dim == other.dim

    @classmethod
    def space_from_vector_obj(cls, vec) -> Self:
        return cls(len(vec))

    @classmethod
    def space_from_dim(cls, dim) -> Self:
        return cls(dim)

    def zero_vector(self) -> NumpyVector:
        return NumpyVector(np.zeros(self.dim))

    def ones_vector(self) -> NumpyVector:
        return NumpyVector(np.ones(self.dim))

    def full_vector(self, value: Complex) -> NumpyVector:
        return NumpyVector(np.full(self.dim, value))

    def make_vector(self, obj) -> NumpyVector:
        obj = np.asarray(obj)
        assert obj.ndim == 1
        assert len(obj) == self.dim
        return NumpyVector(obj)

    def vector_from_numpy(self, data: np.ndarray, ensure_copy=False) -> NumpyVector:
        return self.make_vector(data.copy() if ensure_copy else data)
