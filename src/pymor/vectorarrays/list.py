# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.interfaces import BasicInterface, abstractmethod, abstractclassmethod, classinstancemethod
from pymor.vectorarrays.interfaces import VectorArrayInterface, VectorSpaceInterface, _INDEXTYPES


class VectorInterface(BasicInterface):
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

    @abstractmethod
    def dot(self, other):
        pass

    @abstractmethod
    def l1_norm(self):
        pass

    @abstractmethod
    def l2_norm(self):
        pass

    @abstractmethod
    def l2_norm2(self):
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


class CopyOnWriteVector(VectorInterface):

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
        self._array *= alpha

    def _axpy(self, alpha, x):
        assert self.dim == x.dim
        if alpha == 0:
            return
        if alpha == 1:
            self._array += x._array
        elif alpha == -1:
            self._array -= x._array
        else:
            self._array += x._array * alpha

    def dot(self, other):
        assert self.dim == other.dim
        return np.sum(self._array * other._array)

    def l1_norm(self):
        return np.sum(np.abs(self._array))

    def l2_norm(self):
        return np.linalg.norm(self._array)

    def l2_norm2(self):
        return np.sum((self._array * self._array.conj()).real)

    def dofs(self, dof_indices):
        return self._array[dof_indices]

    def amax(self):
        A = np.abs(self._array)
        max_ind = np.argmax(A)
        max_val = A[max_ind]
        return max_ind, max_val


class ListVectorArray(VectorArrayInterface):
    """|VectorArray| implemented as a Python list of vectors.

    This |VectorArray| implementation is the first choice when
    creating pyMOR wrappers for external solvers which are based
    on single vector objects. In order to do so, a wrapping
    subclass of :class:`VectorInterface` has to be provided
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
        assert isinstance(alpha, _INDEXTYPES) \
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
        assert isinstance(alpha, _INDEXTYPES) \
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

    def dot(self, other):
        assert self.space == other.space
        R = np.empty((len(self._list), len(other)))
        for i, a in enumerate(self._list):
            for j, b in enumerate(other._list):
                R[i, j] = a.dot(b)
        return R

    def pairwise_dot(self, other):
        assert self.space == other.space
        assert len(self._list) == len(other)
        return np.array([a.dot(b) for a, b in zip(self._list, other._list)])

    def gramian(self, product=None):
        if product is not None:
            return super().gramian(product)
        l = len(self._list)
        R = np.empty((l, l))
        for i in range(l):
            for j in range(i, l):
                R[i, j] = self._list[i].dot(self._list[j])
                R[j, i] = R[i, j]
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

    def l1_norm(self):
        return np.array([v.l1_norm() for v in self._list])

    def l2_norm(self):
        return np.array([v.l2_norm() for v in self._list])

    def l2_norm2(self):
        return np.array([v.l2_norm2() for v in self._list])

    def sup_norm(self):
        if self.dim == 0:
            return np.zeros(len(self))
        else:
            return np.array([v.sup_norm() for v in self._list])

    def dofs(self, dof_indices):
        assert isinstance(dof_indices, list) and (len(dof_indices) == 0 or min(dof_indices) >= 0) \
            or (isinstance(dof_indices, np.ndarray) and dof_indices.ndim == 1
                and (len(dof_indices) == 0 or np.min(dof_indices) >= 0))

        R = np.empty((len(self), len(dof_indices)))

        assert len(self) > 0 or len(dof_indices) == 0 or max(dof_indices) < self.dim

        for k, v in enumerate(self._list):
            R[k] = v.dofs(dof_indices)

        return R

    def amax(self):
        assert self.dim > 0

        MI = np.empty(len(self._list), dtype=np.int)
        MV = np.empty(len(self._list))

        for k, v in enumerate(self._list):
            MI[k], MV[k] = v.amax()

        return MI, MV

    def __str__(self):
        return f'ListVectorArray of {len(self._list)} of space {self.space}'


class ListVectorSpace(VectorSpaceInterface):
    """|VectorSpace| of |ListVectorArrays|."""

    dim = None

    @abstractmethod
    def zero_vector(self):
        pass

    @abstractmethod
    def make_vector(self, obj):
        pass

    def vector_from_numpy(self, data, ensure_copy=False):
        raise NotImplementedError

    @classmethod
    def space_from_vector_obj(cls, vec, id_):
        raise NotImplementedError

    @classmethod
    def space_from_dim(cls, dim, id_):
        raise NotImplementedError

    def zeros(self, count=1, reserve=0):
        assert count >= 0 and reserve >= 0
        return ListVectorArray([self.zero_vector() for _ in range(count)], self)

    @classinstancemethod
    def make_array(cls, obj, id_=None):
        if len(obj) == 0:
            raise NotImplementedError
        return cls.space_from_vector_obj(obj[0], id_=id_).make_array(obj)

    @make_array.instancemethod
    def make_array(self, obj):
        return ListVectorArray([self.make_vector(v) for v in obj], self)

    @classinstancemethod
    def from_numpy(cls, data, id_=None, ensure_copy=False):
        return cls.space_from_dim(data.shape[1], id_=id_).from_numpy(data, ensure_copy=ensure_copy)

    @from_numpy.instancemethod
    def from_numpy(self, data, ensure_copy=False):
        return ListVectorArray([self.vector_from_numpy(v, ensure_copy=ensure_copy) for v in data], self)


class NumpyListVectorSpace(ListVectorSpace):

    def __init__(self, dim, id_=None):
        self.dim = dim
        self.id = id_

    def __eq__(self, other):
        return type(other) is NumpyListVectorSpace and self.dim == other.dim and self.id == other.id

    @classmethod
    def space_from_vector_obj(cls, vec, id_):
        return cls(len(vec), id_)

    @classmethod
    def space_from_dim(cls, dim, id_):
        return cls(dim, id_)

    def zero_vector(self):
        return NumpyVector(np.zeros(self.dim))

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
        return self.base[self.base.sub_index(self.ind, ind)]

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

    def __str__(self):
        return f'ListVectorArrayView of {len(self._list)} {str(self.vector_type)}s of dimension {self.dim}'


class ListVectorArrayNumpyView:

    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        return self.array._list[i].to_numpy()

    def __repr__(self):
        return '[' + ',\n '.join(repr(v) for v in self) + ']'
