# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.core.base import BasicObject, abstractmethod, abstractclassmethod, classinstancemethod
from pymor.tools.random import get_random_state
from pymor.vectorarrays.interface import VectorArray, VectorSpace, _create_random_values


class Vector(BasicObject):
    """Interface for vectors used in conjunction with |ListVectorArray|.

    This interface must be satisfied by the individual entries of the
    vector `list` managed by |ListVectorArray|. All interface methods
    have a direct counterpart in the |VectorArray| interface.
    """

    @abstractmethod
    def copy(self, deep=False):
        pass

    @abstractmethod
    def scal(self, alpha):
        pass

    @abstractmethod
    def axpy(self, alpha, x):
        pass

    def inner(self, other):
        raise NotImplementedError

    @abstractmethod
    def norm(self):
        pass

    @abstractmethod
    def norm2(self):
        pass

    def sup_norm(self):
        _, max_val = self.amax()
        return max_val

    @abstractmethod
    def dofs(self, dof_indices):
        pass

    @abstractmethod
    def amax(self):
        pass

    def __add__(self, other):
        result = self.copy()
        result.axpy(1, other)
        return result

    def __iadd__(self, other):
        self.axpy(1, other)
        return self

    __radd__ = __add__

    def __sub__(self, other):
        result = self.copy()
        result.axpy(-1, other)
        return result

    def __isub__(self, other):
        self.axpy(-1, other)
        return self

    def __mul__(self, other):
        result = self.copy()
        result.scal(other)
        return result

    def __imul__(self, other):
        self.scal(other)
        return self

    def __neg__(self):
        result = self.copy()
        result.scal(-1)
        return result

    @property
    def real(self):
        return self.copy()

    @property
    def imag(self):
        return None

    def conj(self):
        return self.copy()


class CopyOnWriteVector(Vector):

    @abstractclassmethod
    def from_instance(cls, instance):
        pass

    @abstractmethod
    def _copy_data(self):
        pass

    @abstractmethod
    def _scal(self, alpha):
        pass

    @abstractmethod
    def _axpy(self, alpha, x):
        pass

    def copy(self, deep=False):
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

    def scal(self, alpha):
        self._copy_data_if_needed()
        self._scal(alpha)

    def axpy(self, alpha, x):
        self._copy_data_if_needed()
        self._axpy(alpha, x)

    def __del__(self):
        try:
            self._refcount[0] -= 1
        except AttributeError:
            pass

    def _copy_data_if_needed(self):
        try:
            if self._refcount[0] > 1:
                self._refcount[0] -= 1
                self._copy_data()
                self._refcount = [1]
        except AttributeError:
            self._refcount = [1]


class ComplexifiedVector(Vector):

    def __init__(self, real_part, imag_part):
        self.real_part, self.imag_part = real_part, imag_part

    def copy(self, deep=False):
        real_part = self.real_part.copy(deep=deep)
        imag_part = self.imag_part.copy(deep=deep) if self.imag_part is not None else None
        return type(self)(real_part, imag_part)

    def scal(self, alpha):
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

    def axpy(self, alpha, x):
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

    def inner(self, other):
        result = self.real_part.inner(other.real_part)
        if self.imag_part is not None:
            result += self.imag_part.inner(other.real_part) * (-1j)
        if other.imag_part is not None:
            result += self.real_part.inner(other.imag_part) * 1j
        if self.imag_part is not None and other.imag_part is not None:
            result += self.imag_part.inner(other.imag_part)
        return result

    def norm(self):
        result = self.real_part.norm()
        if self.imag_part is not None:
            result = np.linalg.norm([result, self.imag_part.norm()])
        return result

    def norm2(self):
        result = self.real_part.norm2()
        if self.imag_part is not None:
            result += self.imag_part.norm2()
        return result

    def sup_norm(self):
        if self.imag_part is not None:
            # we cannot compute the sup_norm from the sup_norms of real_part and imag_part
            return self.amax()[1]
        return self.real_part.sup_norm()

    def dofs(self, dof_indices):
        values = self.real_part.dofs(dof_indices)
        if self.imag_part is not None:
            imag_values = self.imag_part.dofs(dof_indices)
            return values + imag_values * 1j
        else:
            return values

    def amax(self):
        if self.imag_part is not None:
            raise NotImplementedError
        return self.real_part.amax()

    def to_numpy(self, ensure_copy=False):
        if self.imag_part is not None:
            return self.real_part.to_numpy(ensure_copy=False) + self.imag_part.to_numpy(ensure_copy=False) * 1j
        else:
            return self.real_part.to_numpy(ensure_copy=ensure_copy)

    @property
    def real(self):
        return type(self)(self.real_part.copy(), None)

    @property
    def imag(self):
        return type(self)(self.imag_part.copy(), None) if self.imag_part is not None else None

    def conj(self):
        return type(self)(self.real_part.copy(), -self.imag_part if self.imag_part is not None else None)


class NumpyVector(CopyOnWriteVector):
    """Vector stored in a NumPy 1D-array."""

    def __init__(self, array):
        self._array = array

    @classmethod
    def from_instance(cls, instance):
        return cls(instance._array)

    def to_numpy(self, ensure_copy=False):
        if ensure_copy:
            return self._array.copy()
        else:
            self._copy_data_if_needed()
            return self._array

    @property
    def dim(self):
        return len(self._array)

    def _copy_data(self):
        self._array = self._array.copy()

    def _scal(self, alpha):
        try:
            self._array *= alpha
        except TypeError:  # e.g. when scaling real array by complex alpha
            self._array = self._array * alpha

    def _axpy(self, alpha, x):
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

    def inner(self, other):
        assert self.dim == other.dim
        return np.sum(self._array.conj() * other._array)

    def norm(self):
        return np.linalg.norm(self._array)

    def norm2(self):
        return np.sum((self._array * self._array.conj()).real)

    def dofs(self, dof_indices):
        return self._array[dof_indices]

    def amax(self):
        A = np.abs(self._array)
        max_ind = np.argmax(A)
        max_val = A[max_ind]
        return max_ind, np.abs(max_val)

    @property
    def real(self):
        return self.__class__(self._array.real.copy())

    @property
    def imag(self):
        return self.__class__(self._array.imag.copy())

    def conj(self):
        return self.__class__(self._array.conj())


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

    def __init__(self, vectors, space):
        self._list = vectors
        self.space = space

    def to_numpy(self, ensure_copy=False):
        if len(self._list) > 0:
            return np.array([v.to_numpy() for v in self._list])
        else:
            return np.empty((0, self.dim))

    @property
    def _data(self):
        """Return list of NumPy Array views on vector data for hacking / interactive use."""
        return ListVectorArrayNumpyView(self)

    def __len__(self):
        return len(self._list)

    def __getitem__(self, ind):
        if isinstance(ind, Number) and (ind >= len(self) or ind < -len(self)):
            raise IndexError('VectorArray index out of range')
        return ListVectorArrayView(self, ind)

    def __delitem__(self, ind):
        assert self.check_ind(ind)
        if hasattr(ind, '__len__'):
            thelist = self._list
            l = len(thelist)
            remaining = sorted(set(range(l)) - {i if 0 <= i else l+i for i in ind})
            self._list = [thelist[i] for i in remaining]
        else:
            del self._list[ind]

    def append(self, other, remove_from_other=False):
        assert other.space == self.space
        assert not remove_from_other or (other is not self and getattr(other, 'base', None) is not self)

        if not remove_from_other:
            self._list.extend([v.copy() for v in other._list])
        else:
            self._list.extend(other._list)
            if other.is_view:
                del other.base[other.ind]
            else:
                del other[:]

    def copy(self, deep=False):
        return ListVectorArray([v.copy(deep=deep) for v in self._list], self.space)

    def scal(self, alpha):
        assert isinstance(alpha, Number) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (len(self),)

        if type(alpha) is np.ndarray:
            for a, v in zip(alpha, self._list):
                v.scal(a)
        else:
            for v in self._list:
                v.scal(alpha)

    def axpy(self, alpha, x):
        assert self.space == x.space
        len_x = len(x)
        assert len(self) == len_x or len_x == 1
        assert isinstance(alpha, Number) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (len(self),)

        if np.all(alpha == 0):
            return

        if self is x or x.is_view and self is x.base:
            x = x.copy()

        if len(x) == 1:
            xx = x._list[0]
            if type(alpha) is np.ndarray:
                for a, y in zip(alpha, self._list):
                    y.axpy(a, xx)
            else:
                for y in self._list:
                    y.axpy(alpha, xx)
        else:
            if type(alpha) is np.ndarray:
                for a, xx, y in zip(alpha, x._list, self._list):
                    y.axpy(a, xx)
            else:
                for xx, y in zip(x._list, self._list):
                    y.axpy(alpha, xx)

    def inner(self, other, product=None):
        assert self.space == other.space
        if product is not None:
            return product.apply2(self, other)

        return np.array([[a.inner(b) for b in other._list] for a in self._list]).reshape((len(self), len(other)))

    def pairwise_inner(self, other, product=None):
        assert self.space == other.space
        assert len(self._list) == len(other)
        if product is not None:
            return product.pairwise_apply2(self, other)

        return np.array([a.inner(b) for a, b in zip(self._list, other._list)])

    def gramian(self, product=None):
        if product is not None:
            return super().gramian(product)
        l = len(self._list)
        R = [[0.] * l for _ in range(l)]
        for i in range(l):
            for j in range(i, l):
                R[i][j] = self._list[i].inner(self._list[j])
                if i == j:
                    R[i][j] = R[i][j].real
                else:
                    R[j][i] = R[i][j].conjugate()
        R = np.array(R)
        return R

    def lincomb(self, coefficients):
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

    def _norm(self):
        return np.array([v.norm() for v in self._list])

    def _norm2(self):
        return np.array([v.norm2() for v in self._list])

    def sup_norm(self):
        if self.dim == 0:
            return np.zeros(len(self))
        else:
            return np.array([v.sup_norm() for v in self._list])

    def dofs(self, dof_indices):
        assert isinstance(dof_indices, list) and (len(dof_indices) == 0 or min(dof_indices) >= 0) \
            or (isinstance(dof_indices, np.ndarray) and dof_indices.ndim == 1
                and (len(dof_indices) == 0 or np.min(dof_indices) >= 0))
        assert len(self) > 0 or len(dof_indices) == 0 or max(dof_indices) < self.dim
        return np.array([v.dofs(dof_indices) for v in self._list]).reshape((len(self), len(dof_indices)))

    def amax(self):
        assert self.dim > 0

        MI = np.empty(len(self._list), dtype=int)
        MV = np.empty(len(self._list))

        for k, v in enumerate(self._list):
            MI[k], MV[k] = v.amax()

        return MI, MV

    @property
    def real(self):
        return ListVectorArray([v.real for v in self._list], self.space)

    @property
    def imag(self):
        # note that Vector.imag is allowed to return None in case
        # of a real vector, so we have to check for that.
        # returning None is allowed as ComplexifiedVector does not know
        # how to create a new zero vector.
        return ListVectorArray([v.imag or self.space.zero_vector() for v in self._list], self.space)

    def conj(self):
        return self.__class__([v.conj() for v in self._list], self.space)

    def __str__(self):
        return f'{type(self).__name__} of {len(self._list)} vectors of space {self.space}'


class ListVectorSpace(VectorSpace):
    """|VectorSpace| of |ListVectorArrays|."""

    dim = None
    vector_type = Vector

    @abstractmethod
    def zero_vector(self):
        pass

    def ones_vector(self):
        return self.full_vector(1.)

    def full_vector(self, value):
        return self.vector_from_numpy(np.full(self.dim, value))

    def random_vector(self, distribution, random_state, **kwargs):
        values = _create_random_values(self.dim, distribution, random_state, **kwargs)
        return self.vector_from_numpy(values)

    @abstractmethod
    def make_vector(self, obj):
        pass

    def vector_from_numpy(self, data, ensure_copy=False):
        raise NotImplementedError

    @classmethod
    def space_from_vector_obj(cls, vec, id):
        raise NotImplementedError

    @classmethod
    def space_from_dim(cls, dim, id):
        raise NotImplementedError

    def zeros(self, count=1, reserve=0):
        assert count >= 0 and reserve >= 0
        return ListVectorArray([self.zero_vector() for _ in range(count)], self)

    def ones(self, count=1, reserve=0):
        assert count >= 0 and reserve >= 0
        return ListVectorArray([self.ones_vector() for _ in range(count)], self)

    def full(self, value, count=1, reserve=0):
        assert count >= 0 and reserve >= 0
        return ListVectorArray([self.full_vector(value) for _ in range(count)], self)

    def random(self, count=1, distribution='uniform', random_state=None, seed=None, reserve=0, **kwargs):
        assert count >= 0 and reserve >= 0
        assert random_state is None or seed is None
        random_state = get_random_state(random_state, seed)
        return ListVectorArray([self.random_vector(distribution=distribution, random_state=random_state, **kwargs)
                                for _ in range(count)], self)

    @classinstancemethod
    def make_array(cls, obj, id=None):
        if len(obj) == 0:
            raise NotImplementedError
        return cls.space_from_vector_obj(obj[0], id=id).make_array(obj)

    @make_array.instancemethod
    def make_array(self, obj):
        """:noindex:"""
        return ListVectorArray([v if isinstance(v, self.vector_type) else self.make_vector(v) for v in obj], self)

    @classinstancemethod
    def from_numpy(cls, data, id=None, ensure_copy=False):
        return cls.space_from_dim(data.shape[1], id=id).from_numpy(data, ensure_copy=ensure_copy)

    @from_numpy.instancemethod
    def from_numpy(self, data, ensure_copy=False):
        """:noindex:"""
        return ListVectorArray([self.vector_from_numpy(v, ensure_copy=ensure_copy) for v in data], self)


class ComplexifiedListVectorSpace(ListVectorSpace):

    real_vector_type = Vector
    vector_type = ComplexifiedVector

    @abstractmethod
    def real_zero_vector(self):
        pass

    def zero_vector(self):
        return self.vector_type(self.real_zero_vector(), None)

    def real_full_vector(self, value):
        return self.real_vector_from_numpy(np.full(self.dim, value))

    def full_vector(self, value):
        return self.vector_type(self.real_full_vector(value), None)

    def real_random_vector(self, distribution, random_state, **kwargs):
        values = _create_random_values(self.dim, distribution, random_state, **kwargs)
        return self.real_vector_from_numpy(values)

    def random_vector(self, distribution, random_state, **kwargs):
        return self.vector_type(self.real_random_vector(distribution, random_state, **kwargs), None)

    @abstractmethod
    def real_make_vector(self, obj):
        pass

    def make_vector(self, obj):
        if isinstance(obj, self.real_vector_type):
            return self.vector_type(obj, None)
        else:
            return self.vector_type(self.real_make_vector(obj), None)

    def real_vector_from_numpy(self, data, ensure_copy=False):
        raise NotImplementedError

    def vector_from_numpy(self, data, ensure_copy=False):
        if np.iscomplexobj(data):
            real_part = self.real_vector_from_numpy(data.real)
            imag_part = self.real_vector_from_numpy(data.imag)
        else:
            real_part = self.real_vector_from_numpy(data, ensure_copy=ensure_copy)
            imag_part = None
        return self.vector_type(real_part, imag_part)


class NumpyListVectorSpace(ListVectorSpace):

    vector_type = NumpyVector

    def __init__(self, dim, id=None):
        self.dim = dim
        self.id = id

    def __eq__(self, other):
        return type(other) is NumpyListVectorSpace and self.dim == other.dim and self.id == other.id

    @classmethod
    def space_from_vector_obj(cls, vec, id):
        return cls(len(vec), id)

    @classmethod
    def space_from_dim(cls, dim, id):
        return cls(dim, id)

    def zero_vector(self):
        return NumpyVector(np.zeros(self.dim))

    def ones_vector(self):
        return NumpyVector(np.ones(self.dim))

    def full_vector(self, value):
        return NumpyVector(np.full(self.dim, value))

    def make_vector(self, obj):
        obj = np.asarray(obj)
        assert obj.ndim == 1 and len(obj) == self.dim
        return NumpyVector(obj)

    def vector_from_numpy(self, data, ensure_copy=False):
        return self.make_vector(data.copy() if ensure_copy else data)


class ListVectorArrayView(ListVectorArray):

    is_view = True

    def __init__(self, base, ind):
        self.base = base
        assert base.check_ind(ind)
        self.ind = base.normalize_ind(ind)
        if type(ind) is slice:
            self._list = base._list[ind]
        elif hasattr(ind, '__len__'):
            _list = base._list
            self._list = [_list[i] for i in ind]
        else:
            self._list = [base._list[ind]]

    @property
    def space(self):
        return self.base.space

    def __getitem__(self, ind):
        try:
            return self.base[self.base.sub_index(self.ind, ind)]
        except IndexError:
            raise IndexError('VectorArray index out of range')

    def __delitem__(self, ind):
        raise TypeError('Cannot remove from ListVectorArrayView')

    def append(self, other, remove_from_other=False):
        raise TypeError('Cannot append to ListVectorArrayView')

    def scal(self, alpha):
        assert self.base.check_ind_unique(self.ind)
        super().scal(alpha)

    def axpy(self, alpha, x):
        assert self.base.check_ind_unique(self.ind)
        if x is self.base or x.is_view and x.base is self.base:
            x = x.copy()
        super().axpy(alpha, x)


class ListVectorArrayNumpyView:

    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        return self.array._list[i].to_numpy()

    def __repr__(self):
        return '[' + ',\n '.join(repr(v) for v in self) + ']'
