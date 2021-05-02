# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import annotations

from numbers import Number
from typing import Tuple, Optional, List, Any, Sequence, TYPE_CHECKING, Type, cast

import numpy as np
from numpy import ndarray
from numpy.random import RandomState

from pymor.core.base import BasicObject, abstractmethod, abstractclassmethod, classinstancemethod
from pymor.tools.random import get_random_state
from pymor.typing import RealOrComplex, Real, ScalCoeffs, Index, ScalarIndex, SCALAR_INDICES
from pymor.vectorarrays.interface import VectorArray, VectorSpace, _create_random_values

if TYPE_CHECKING:
    from pymor.operators.interface import Operator


class Vector(BasicObject):
    """Interface for vectors used in conjunction with |ListVectorArray|.

    This interface must be satisfied by the individual entries of the
    vector `list` managed by |ListVectorArray|. All interface methods
    have a direct counterpart in the |VectorArray| interface.
    """

    @abstractmethod
    def copy(self, deep: bool = False) -> Vector:
        pass

    @abstractmethod
    def scal(self, alpha: RealOrComplex) -> None:
        pass

    @abstractmethod
    def axpy(self, alpha: RealOrComplex, x: Vector) -> None:
        pass

    def inner(self, other: Vector) -> RealOrComplex:
        raise NotImplementedError

    @abstractmethod
    def norm(self) -> float:
        pass

    @abstractmethod
    def norm2(self) -> float:
        pass

    def sup_norm(self) -> float:
        _, max_val = self.amax()
        return max_val

    @abstractmethod
    def dofs(self, dof_indices: ndarray) -> ndarray:
        pass

    @abstractmethod
    def amax(self) -> Tuple[int, float]:
        pass

    def to_numpy(self, ensure_copy: bool = False) -> ndarray:
        raise NotImplementedError

    def __add__(self, other: Vector) -> Vector:
        result = self.copy()
        result.axpy(1, other)
        return result

    def __iadd__(self, other: Vector) -> Vector:
        self.axpy(1, other)
        return self

    __radd__ = __add__

    def __sub__(self, other: Vector) -> Vector:
        result = self.copy()
        result.axpy(-1, other)
        return result

    def __isub__(self, other: Vector) -> Vector:
        self.axpy(-1, other)
        return self

    def __mul__(self, other: RealOrComplex) -> Vector:
        result = self.copy()
        result.scal(other)
        return result

    def __imul__(self, other: RealOrComplex) -> Vector:
        self.scal(other)
        return self

    def __neg__(self) -> Vector:
        result = self.copy()
        result.scal(-1)
        return result

    @property
    def real(self) -> Vector:
        return self.copy()

    @property
    def imag(self) -> Optional[Vector]:
        return None

    def conj(self) -> Vector:
        return self.copy()


class CopyOnWriteVector(Vector):

    _refcount: List[int]

    @abstractclassmethod
    def from_instance(cls, instance: CopyOnWriteVector) -> CopyOnWriteVector:
        pass

    @abstractmethod
    def _copy_data(self) -> None:
        pass

    @abstractmethod
    def _scal(self, alpha: RealOrComplex) -> None:
        pass

    @abstractmethod
    def _axpy(self, alpha: RealOrComplex, x: CopyOnWriteVector) -> None:
        pass

    def copy(self, deep: bool = False) -> CopyOnWriteVector:
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

    def scal(self, alpha: RealOrComplex) -> None:
        self._copy_data_if_needed()
        self._scal(alpha)

    def axpy(self, alpha: RealOrComplex, x: Vector) -> None:
        assert isinstance(x, type(self))
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

    real_part: Vector
    imag_part: Optional[Vector]

    def __init__(self, real_part: Vector, imag_part: Optional[Vector]):
        self.real_part, self.imag_part = real_part, imag_part

    def copy(self, deep: bool = False) -> ComplexifiedVector:
        real_part = self.real_part.copy(deep=deep)
        imag_part = self.imag_part.copy(deep=deep) if self.imag_part is not None else None
        return type(self)(real_part, imag_part)

    def scal(self, alpha: RealOrComplex) -> None:
        if self.imag_part is None:
            self.real_part.scal(alpha.real)
            if alpha.imag != 0:
                self.imag_part = self.real_part * alpha.imag
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

    def axpy(self, alpha: RealOrComplex, x: Vector) -> None:
        assert isinstance(x, ComplexifiedVector)
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

    def inner(self, other: Vector) -> RealOrComplex:
        assert isinstance(other, ComplexifiedVector)
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

    def dofs(self, dof_indices: ndarray) -> ndarray:
        values = self.real_part.dofs(dof_indices)
        if self.imag_part is not None:
            imag_values = self.imag_part.dofs(dof_indices)
            return values + imag_values * 1j
        else:
            return values

    def amax(self) -> Tuple[int, float]:
        if self.imag_part is not None:
            raise NotImplementedError
        return self.real_part.amax()

    def to_numpy(self, ensure_copy: bool = False) -> ndarray:
        if self.imag_part is not None:
            return self.real_part.to_numpy(ensure_copy=False) + self.imag_part.to_numpy(ensure_copy=False) * 1j
        else:
            return self.real_part.to_numpy(ensure_copy=ensure_copy)

    @property
    def real(self) -> ComplexifiedVector:
        return type(self)(self.real_part.copy(), None)

    @property
    def imag(self) -> Optional[ComplexifiedVector]:
        return type(self)(self.imag_part.copy(), None) if self.imag_part is not None else None

    def conj(self) -> ComplexifiedVector:
        return type(self)(self.real_part.copy(), -self.imag_part if self.imag_part is not None else None)


class NumpyVector(CopyOnWriteVector):
    """Vector stored in a NumPy 1D-array."""

    _array: ndarray

    def __init__(self, array: ndarray):
        self._array = array

    @classmethod
    def from_instance(cls, instance: CopyOnWriteVector) -> NumpyVector:
        assert isinstance(instance, NumpyVector)
        return cls(instance._array)

    def to_numpy(self, ensure_copy: bool = False) -> ndarray:
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

    def _scal(self, alpha: RealOrComplex) -> None:
        try:
            self._array *= alpha
        except TypeError:  # e.g. when scaling real array by complex alpha
            self._array = self._array * alpha

    def _axpy(self, alpha: RealOrComplex, x: CopyOnWriteVector) -> None:
        assert isinstance(x, NumpyVector)
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

    def inner(self, other: Vector) -> RealOrComplex:
        assert isinstance(other, NumpyVector)
        assert self.dim == other.dim
        return np.sum(self._array.conj() * other._array)

    def norm(self) -> float:
        return np.linalg.norm(self._array)

    def norm2(self) -> float:
        return np.sum((self._array * self._array.conj()).real)

    def dofs(self, dof_indices: ndarray) -> ndarray:
        return self._array[dof_indices]

    def amax(self) -> Tuple[int, float]:
        A = np.abs(self._array)
        max_ind = np.argmax(A)
        max_val = A[max_ind]
        return max_ind, np.abs(max_val)

    @property
    def real(self) -> NumpyVector:
        return type(self)(self._array.real.copy())

    @property
    def imag(self) -> NumpyVector:
        return type(self)(self._array.imag.copy())

    def conj(self) -> NumpyVector:
        return type(self)(self._array.conj())


class ListVectorArray(VectorArray):
    """|VectorArray| implemented as a Python list of vectors.

    This |VectorArray| implementation is the first choice when
    creating pyMOR wrappers for external solvers which are based
    on single vector objects. In order to do so, a wrapping
    subclass of :class:`Vector` has to be provided
    on which the implementation of |ListVectorArray| will operate.
    The associated |VectorSpace| is a subclass of
    :class:`ListVectorSpace`.

    For an example, see :class:`NumpyVector`, :class:`NumpyListVectorSpace`
    or :class:`~pymor.bindings.fenics.FenicsVector`,
    :class:`~pymor.bindings.fenics.FenicsVectorSpace`.
    """

    _NONE = ()
    space: ListVectorSpace

    def __init__(self, vectors: List[Vector], space: ListVectorSpace):
        self._list = vectors
        self.space = space

    def to_numpy(self, ensure_copy: bool = False) -> ndarray:
        if len(self._list) > 0:
            return np.array([v.to_numpy() for v in self._list])
        else:
            return np.empty((0, self.dim))

    @property
    def _data(self) -> ListVectorArrayNumpyView:
        """Return list of NumPy Array views on vector data for hacking / interactive use."""
        return ListVectorArrayNumpyView(self)

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, ind: Index) -> ListVectorArrayView:
        if isinstance(ind, SCALAR_INDICES) and (ind >= len(self) or ind < -len(self)):
            raise IndexError('VectorArray index out of range')
        return ListVectorArrayView(self, ind)

    def __delitem__(self, ind: Index) -> None:
        assert self.check_ind(ind)
        if isinstance(ind, (list, ndarray)):
            thelist = self._list
            l = len(thelist)
            remaining = sorted(set(range(l)) - {i if 0 <= i else l+i for i in ind})
            self._list = [thelist[i] for i in remaining]
        else:
            del self._list[ind]

    def append(self, other: VectorArray, remove_from_other: bool = False) -> None:
        assert isinstance(other, ListVectorArray) and other in self.space
        assert not remove_from_other or (other is not self and getattr(other, 'base', None) is not self)

        if not remove_from_other:
            self._list.extend([v.copy() for v in other._list])
        else:
            self._list.extend(other._list)
            if isinstance(other, ListVectorArrayView):
                del other.base[other.ind]
            else:
                del other[:]

    def copy(self, deep: bool = False) -> ListVectorArray:
        return ListVectorArray([v.copy(deep=deep) for v in self._list], self.space)

    def scal(self, alpha: ScalCoeffs) -> None:
        assert isinstance(alpha, Number) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (len(self),)

        if isinstance(alpha, ndarray):
            for a, v in zip(alpha, self._list):
                v.scal(a)
        else:
            for v in self._list:
                v.scal(alpha)

    def axpy(self, alpha: ScalCoeffs, x: VectorArray) -> None:
        assert isinstance(x, ListVectorArray) and x in self.space
        len_x = len(x)
        assert len(self) == len_x or len_x == 1
        assert isinstance(alpha, Number) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (len(self),)

        if np.all(alpha == 0):
            return

        if self is x or isinstance(x, ListVectorArrayView) and self is x.base:
            x = x.copy()

        if len(x) == 1:
            xx = x._list[0]
            if isinstance(alpha, ndarray):
                for a, y in zip(alpha, self._list):
                    y.axpy(a, xx)
            else:
                for y in self._list:
                    y.axpy(alpha, xx)
        else:
            if isinstance(alpha, ndarray):
                for a, xx, y in zip(alpha, x._list, self._list):
                    y.axpy(a, xx)
            else:
                for xx, y in zip(x._list, self._list):
                    y.axpy(alpha, xx)

    def inner(self, other: VectorArray, product: Optional[Operator] = None) -> ndarray:
        assert isinstance(other, ListVectorArray) and other in self.space
        if product is not None:
            return product.apply2(self, other)

        return np.array([[a.inner(b) for b in other._list] for a in self._list]).reshape((len(self), len(other)))

    def pairwise_inner(self, other: VectorArray, product: Optional[Operator] = None) -> ndarray:
        assert isinstance(other, ListVectorArray) and other in self.space
        assert len(self._list) == len(other)
        if product is not None:
            return product.pairwise_apply2(self, other)

        return np.array([a.inner(b) for a, b in zip(self._list, other._list)])

    def gramian(self, product: Optional[Operator] = None) -> ndarray:
        if product is not None:
            return super().gramian(product)
        l = len(self._list)
        R: List[List[RealOrComplex]] = [[0.] * l for _ in range(l)]
        for i in range(l):
            for j in range(i, l):
                R[i][j] = self._list[i].inner(self._list[j])
                if i == j:
                    R[i][j] = R[i][j].real
                else:
                    R[j][i] = R[i][j].conjugate()
        R = np.array(R)
        return R

    def lincomb(self, coefficients: ndarray) -> ListVectorArray:
        assert 1 <= coefficients.ndim <= 2
        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, :]

        assert coefficients.shape[1] == len(self)

        RL = []
        for coeffs in coefficients:
            R = self.space.zero_vector()
            for v, c in zip(self._list, coeffs):
                R.axpy(c, v)
            RL.append(R)

        return ListVectorArray(RL, self.space)

    def _norm(self) -> ndarray:
        return np.array([v.norm() for v in self._list])

    def _norm2(self) -> ndarray:
        return np.array([v.norm2() for v in self._list])

    def sup_norm(self) -> ndarray:
        if self.dim == 0:
            return np.zeros(len(self))
        else:
            return np.array([v.sup_norm() for v in self._list])

    def dofs(self, dof_indices: ndarray) -> ndarray:
        assert isinstance(dof_indices, list) and (len(dof_indices) == 0 or min(dof_indices) >= 0) \
            or (isinstance(dof_indices, np.ndarray) and dof_indices.ndim == 1
                and (len(dof_indices) == 0 or np.min(dof_indices) >= 0))
        assert len(self) > 0 or len(dof_indices) == 0 or max(dof_indices) < self.dim
        return np.array([v.dofs(dof_indices) for v in self._list]).reshape((len(self), len(dof_indices)))

    def amax(self) -> Tuple[ndarray, ndarray]:
        assert self.dim > 0

        MI = np.empty(len(self._list), dtype=np.int)
        MV = np.empty(len(self._list))

        for k, v in enumerate(self._list):
            MI[k], MV[k] = v.amax()

        return MI, MV

    @property
    def real(self) -> ListVectorArray:
        return ListVectorArray([v.real for v in self._list], self.space)

    @property
    def imag(self) -> ListVectorArray:
        # note that Vector.imag is allowed to return None in case
        # of a real vector, so we have to check for that.
        # returning None is allowed as ComplexifiedVector does not know
        # how to create a new zero vector.
        return ListVectorArray([v.imag or self.space.zero_vector() for v in self._list], self.space)

    def conj(self) -> ListVectorArray:
        return type(self)([v.conj() for v in self._list], self.space)

    def __str__(self) -> str:
        return f'{type(self).__name__} of {len(self._list)} vectors of space {self.space}'


class ListVectorSpace(VectorSpace):
    """|VectorSpace| of |ListVectorArrays|."""

    @abstractmethod
    def zero_vector(self) -> Vector:
        pass

    def ones_vector(self) -> Vector:
        return self.full_vector(1.)

    def full_vector(self, value) -> Vector:
        return self.vector_from_numpy(np.full(self.dim, value))

    def random_vector(self, distribution: str, random_state: RandomState, **kwargs) -> Vector:
        values = _create_random_values((self.dim,), distribution, random_state, **kwargs)
        return self.vector_from_numpy(values)

    @abstractmethod
    def make_vector(self, obj: Any) -> Vector:
        pass

    def vector_from_numpy(self, data: ndarray, ensure_copy: bool = False) -> Vector:
        raise NotImplementedError

    @classmethod
    def space_from_vector_obj(cls, obj: Any, id: Optional[str]) -> ListVectorSpace:
        raise NotImplementedError

    @classmethod
    def space_from_dim(cls, dim: int, id: Optional[str]) -> ListVectorSpace:
        raise NotImplementedError

    def zeros(self, count: int = 1, reserve: int = 0) -> ListVectorArray:
        assert count >= 0 and reserve >= 0
        return ListVectorArray([self.zero_vector() for _ in range(count)], self)

    def ones(self, count: int = 1, reserve: int = 0) -> ListVectorArray:
        assert count >= 0 and reserve >= 0
        return ListVectorArray([self.ones_vector() for _ in range(count)], self)

    def full(self, value: RealOrComplex, count: int = 1, reserve: int = 0) -> ListVectorArray:
        assert count >= 0 and reserve >= 0
        return ListVectorArray([self.full_vector(value) for _ in range(count)], self)

    def random(self, count: int = 1, distribution: str = 'uniform', random_state: Optional[RandomState] = None,
               seed: Optional[int] = None, reserve: int = 0, **kwargs) -> ListVectorArray:
        assert count >= 0 and reserve >= 0
        assert random_state is None or seed is None
        random_state = get_random_state(random_state, seed)
        return ListVectorArray([self.random_vector(distribution=distribution, random_state=random_state, **kwargs)
                                for _ in range(count)], self)

    @classinstancemethod
    def make_array(cls, obj: Sequence, id: Optional[str] = None) -> ListVectorArray:
        if len(obj) == 0:
            raise NotImplementedError
        return cls.space_from_vector_obj(obj[0], id=id).make_array(obj)

    @make_array.instancemethod  # type: ignore[no-redef]
    def make_array(self, obj: Sequence) -> ListVectorArray:
        return ListVectorArray([v if isinstance(v, Vector) else self.make_vector(v) for v in obj], self)

    @classinstancemethod
    def from_numpy(cls, data: ndarray, id: Optional[str] = None, ensure_copy: bool = False) -> ListVectorArray:
        return cls.space_from_dim(data.shape[1], id=id).from_numpy(data, ensure_copy=ensure_copy)

    @from_numpy.instancemethod  # type: ignore[no-redef]
    def from_numpy(self, data: ndarray, ensure_copy: bool = False) -> ListVectorArray:
        return ListVectorArray([self.vector_from_numpy(v, ensure_copy=ensure_copy) for v in data], self)


class ComplexifiedListVectorSpace(ListVectorSpace):

    complexified_vector_type: Type[ComplexifiedVector] = ComplexifiedVector

    @abstractmethod
    def real_zero_vector(self) -> Vector:
        pass

    def zero_vector(self) -> Vector:
        return self.complexified_vector_type(self.real_zero_vector(), None)

    def real_full_vector(self, value: Real) -> Vector:
        return self.real_vector_from_numpy(np.full(self.dim, value))

    def full_vector(self, value: RealOrComplex) -> ComplexifiedVector:
        return self.complexified_vector_type(self.real_full_vector(value), None)

    def real_random_vector(self, distribution: str, random_state: RandomState, **kwargs) -> Vector:
        values = _create_random_values((self.dim,), distribution, random_state, **kwargs)
        return self.real_vector_from_numpy(values)

    def random_vector(self, distribution: str, random_state: RandomState, **kwargs) -> ComplexifiedVector:
        return self.complexified_vector_type(self.real_random_vector(distribution, random_state, **kwargs), None)

    @abstractmethod
    def real_make_vector(self, obj: Any) -> Vector:
        pass

    def make_vector(self, obj: Any) -> ComplexifiedVector:
        return self.complexified_vector_type(self.real_make_vector(obj), None)

    def real_vector_from_numpy(self, data: ndarray, ensure_copy: bool = False) -> Vector:
        raise NotImplementedError

    def vector_from_numpy(self, data: ndarray, ensure_copy: bool = False) -> ComplexifiedVector:

        imag_part: Optional[Vector]

        if np.iscomplexobj(data):
            real_part = self.real_vector_from_numpy(data.real)
            imag_part = self.real_vector_from_numpy(data.imag)
        else:
            real_part = self.real_vector_from_numpy(data, ensure_copy=ensure_copy)
            imag_part = None
        return self.complexified_vector_type(real_part, imag_part)


class NumpyListVectorSpace(ListVectorSpace):

    def __init__(self, dim: int, id: Optional[str] = None) -> None:
        self.dim = dim
        self.id = id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NumpyListVectorSpace) and self.dim == other.dim and self.id == other.id

    @classmethod
    def space_from_vector_obj(cls, vec: ndarray, id: Optional[str]) -> NumpyListVectorSpace:
        return cls(len(vec), id)

    @classmethod
    def space_from_dim(cls, dim: int, id: Optional[str]) -> NumpyListVectorSpace:
        return cls(dim, id)

    def zero_vector(self) -> NumpyVector:
        return NumpyVector(np.zeros(self.dim))

    def ones_vector(self) -> NumpyVector:
        return NumpyVector(np.ones(self.dim))

    def full_vector(self, value) -> NumpyVector:
        return NumpyVector(np.full(self.dim, value))

    def make_vector(self, obj: ndarray) -> NumpyVector:
        obj = np.asarray(obj)
        assert obj.ndim == 1 and len(obj) == self.dim
        return NumpyVector(obj)

    def vector_from_numpy(self, data: ndarray, ensure_copy: bool = False) -> NumpyVector:
        return self.make_vector(data.copy() if ensure_copy else data)


class ListVectorArrayView(ListVectorArray):

    is_view: bool = True

    def __init__(self, base: ListVectorArray, ind: Index) -> None:
        self.base = base
        assert base.check_ind(ind)
        self.ind = base.normalize_ind(ind)
        if isinstance(ind, slice):
            self._list = base._list[ind]
        elif isinstance(ind, (list, ndarray)):
            _list = base._list
            self._list = [_list[i] for i in ind]
        else:
            self._list = [base._list[ind]]

    @property
    def space(self) -> ListVectorSpace:  # type: ignore[override]
        return self.base.space

    def __getitem__(self, ind: Index) -> ListVectorArrayView:
        try:
            return self.base[self.base.sub_index(self.ind, ind)]
        except IndexError:
            raise IndexError('VectorArray index out of range')

    def __delitem__(self, ind: Index) -> None:
        raise TypeError('Cannot remove from ListVectorArrayView')

    def append(self, other: VectorArray, remove_from_other: bool = False) -> None:
        raise TypeError('Cannot append to ListVectorArrayView')

    def scal(self, alpha: ScalCoeffs) -> None:
        assert self.base.check_ind_unique(self.ind)
        super().scal(alpha)

    def axpy(self, alpha: ScalCoeffs, x: VectorArray) -> None:
        assert self.base.check_ind_unique(self.ind)
        if x is self.base or isinstance(x, ListVectorArrayView) and x.base is self.base:
            x = x.copy()
        super().axpy(alpha, x)


class ListVectorArrayNumpyView:

    def __init__(self, array: ndarray) -> None:
        self.array = array

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, i: ScalarIndex) -> ndarray:
        return self.array._list[i].to_numpy()

    def __repr__(self) -> str:
        return '[' + ',\n '.join(repr(v) for v in cast(Sequence, self)) + ']'
