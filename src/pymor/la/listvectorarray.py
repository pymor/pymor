# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip
from numbers import Number

import numpy as np

from pymor.core.interfaces import BasicInterface, abstractmethod, abstractclassmethod, abstractproperty
from pymor.la.interfaces import VectorArrayInterface
from pymor.tools import float_cmp_all


class VectorInterface(BasicInterface):
    '''Interface for vectors.

    This Interface is mainly intended to be used in conjunction with |ListVectorArray|. In general, all
    pyMOR objects operate on |VectorArrays| instead of single vectors! All methods of the interface have
    a direct counterpart in the |VectorArray| interface.
    '''

    @abstractclassmethod
    def zeros(cls, dim):
        pass

    @abstractproperty
    def dim(self):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def almost_equal(self, other, rtol=None, atol=None):
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

    def sup_norm(self):
        if self.dim == 0:
            return 0.
        else:
            _, max_val = self.amax()
            return max_val

    @abstractmethod
    def components(self, component_indices):
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


class NumpyVector(VectorInterface):
    '''Vector stored in a NumPy 1D-array.'''

    def __init__(self, instance, dtype=None, copy=False, order=None, subok=False):
        if isinstance(instance, np.ndarray) and not copy:
            self._array = instance
        else:
            self._array = np.array(instance, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=1)
        assert self._array.ndim == 1

    @property
    def data(self):
        return self._array

    def zeros(cls, dim):
        return NumpyVector(np.zeros(dim))

    @property
    def dim(self):
        return len(self._array)

    def copy(self):
        return NumpyVector(self._array, copy=True)

    def almost_equal(self, other, rtol=None, atol=None):
        assert self.dim == other.dim
        return float_cmp_all(self._array, other._array, rtol=rtol, atol=atol)

    def scal(self, alpha):
        self._array *= alpha

    def axpy(self, alpha, x):
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
        return np.sum(np.power(self._array, 2))**(1/2)

    def components(self, component_indices):
        return self._array[component_indices]

    def amax(self):
        A = np.abs(self._array)
        max_ind = np.argmax(A)
        max_val = A[max_ind]
        return (max_ind, max_val)


class ListVectorArray(VectorArrayInterface):
    '''|VectorArray| implementation via a python list of vectors.

    In order to create a |ListVectorArray|, derive from this
    class and define the `vector_type` class attribute.

    Attributes
    ----------
    vector_type
        The type of of :class:`Vectors <VectorInterface>` to store.
        Each vector has to be of the same type.
    '''

    def __init__(self, vectors, dim=None, copy=True):
        assert dim is None or dim >= 0
        if not copy:
            if isinstance(vectors, list):
                self._list = vectors
            else:
                self._list = list(vectors)
        else:
            self._list = [v.copy() for v in vectors]
        if dim is None:
            assert len(self._list) > 0
            dim = self._list[0].dim
        self._dim = dim
        assert all(v.dim == dim for v in self._list)

    @classmethod
    def empty(cls, dim, reserve=0):
        assert reserve >= 0
        return cls([], dim, copy=False)

    @classmethod
    def zeros(cls, dim, count=1):
        assert count >= 0
        return cls([cls.vector_type.zeros(dim) for c in xrange(count)], dim=dim, copy=False)

    def __len__(self):
        return len(self._list)

    @property
    def data(self):
        if not hasattr(self.vector_type, 'data'):
            raise TypeError('{} does not have a data attribute'.format(self.vector_type))
        if len(self._list) > 0:
            return np.array([v.data for v in self._list])
        else:
            return np.empty((0, self._dim))

    @property
    def dim(self):
        return self._dim

    def copy(self, ind=None):
        assert self.check_ind(ind)

        if ind is None:
            return type(self)(self._list, dim=self._dim, copy=True)
        elif isinstance(ind, Number):
            return type(self)([self._list[ind]], copy=True)
        else:
            return type(self)([self._list[i] for i in ind], dim=self._dim, copy=True)

    def append(self, other, o_ind=None, remove_from_other=False):
        assert other.check_ind(o_ind)
        assert other.dim == self.dim
        assert other is not self or not remove_from_other

        other_list = other._list
        if not remove_from_other:
            if o_ind is None:
                self._list.extend([v.copy() for v in other_list])
            elif isinstance(o_ind, Number):
                self._list.append(other_list[o_ind].copy())
            else:
                self._list.extend([other_list[i].copy() for i in o_ind])
        else:
            if o_ind is None:
                self._list.extend(other_list)
                other._list = []
            elif isinstance(o_ind, Number):
                self._list.append(other_list.pop(o_ind))
            else:
                self._list.extend([other_list[i] for i in o_ind])
                remaining = sorted(set(xrange(len(other_list))) - set(o_ind))
                other._list = [other_list[i] for i in remaining]

    def remove(self, ind=None):
        assert self.check_ind(ind)
        if ind is None:
            self._list = []
        elif isinstance(ind, Number):
            del self._list[ind]
        else:
            thelist = self._list
            remaining = sorted(set(xrange(len(self))) - set(ind))
            self._list = [thelist[i] for i in remaining]

    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        assert self.check_ind_unique(ind)
        assert other.check_ind(o_ind)
        assert other.dim == self.dim
        assert other is not self or not remove_from_other

        if ind is None:
            c = type(self).empty(self.dim)
            c.append(other, o_ind=o_ind, remove_from_other=remove_from_other)
            assert len(c) == len(self)
            self._list = c._list
        elif isinstance(ind, Number):
            if o_ind is None:
                assert len(other._list) == 1
                if not remove_from_other:
                    self._list[ind] = other._list[0].copy()
                else:
                    self._list[ind] = other._list.pop()
            else:
                if not isinstance(o_ind, Number):
                    assert len(o_ind) == 1
                    o_ind = o_ind[0]
                if not remove_from_other:
                    self._list[ind] = other._list[o_ind].copy()
                else:
                    self._list[ind] = other._list.pop(o_ind)
        else:
            if isinstance(o_ind, Number):
                assert len(ind) == 1
                if not remove_from_other:
                    self._list[ind[0]] = other._list[o_ind].copy()
                else:
                    self._list[ind[0]] = other._list.pop(o_ind)
            else:
                o_ind = xrange(len(other)) if o_ind is None else o_ind
                assert len(ind) == len(o_ind)
                if not remove_from_other:
                    l = self._list
                    # if other is self, we have to make a copy of our list, to prevent
                    # messing things up, e.g. when swapping vectors
                    other_list = list(l) if other is self else other._list
                    for i, oi in izip(ind, o_ind):
                        l[i] = other_list[oi].copy()
                else:
                    for i, oi in izip(ind, o_ind):
                        self._list[i] = other._list[oi]
                    other_list = other._list
                    remaining = sorted(set(xrange(len(other_list))) - set(o_ind))
                    other._list = [other_list[i] for i in remaining]

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        assert self.check_ind(ind)
        assert other.check_ind(o_ind)
        assert other.dim == self.dim

        if ind is None:
            ind = xrange(len(self._list))
        elif isinstance(ind, Number):
            ind = [ind]

        if o_ind is None:
            o_ind = xrange(len(other._list))
        elif isinstance(o_ind, Number):
            o_ind = [o_ind]

        l = self._list
        ol = other._list

        if len(ind) == 1:
            a = l[ind[0]]
            return np.array([a.almost_equal(ol[oi], rtol=rtol, atol=atol) for oi in o_ind])
        elif len(o_ind) == 1:
            b = ol[o_ind[0]]
            return np.array([l[i].almost_equal(b, rtol=rtol, atol=atol) for i in ind])
        else:
            assert len(ind) == len(o_ind)
            return np.array([l[i].almost_equal(ol[oi], rtol=rtol, atol=atol) for i, oi in izip(ind, o_ind)])

    def scal(self, alpha, ind=None):
        assert self.check_ind_unique(ind)
        assert isinstance(alpha, Number)

        if ind is None:
            for v in self._list:
                v.scal(alpha)
        elif isinstance(ind, Number):
            self._list[ind].scal(alpha)
        else:
            l = self._list
            for i in ind:
                l[i].scal(alpha)

    def axpy(self, alpha, x, ind=None, x_ind=None):
        assert self.check_ind_unique(ind)
        assert x.check_ind(x_ind)
        assert self.dim == x.dim
        assert self.len_ind(ind) == x.len_ind(x_ind)

        if self is x:
            if ind is None or x_ind is None:
                self.axpy(alpha, x.copy(), ind, x_ind)
                return
            ind_set = {ind} if isinstance(ind, Number) else set(ind)
            x_ind_set = {x_ind} if isinstance(x_ind, Number) else set(x_ind)
            if ind_set.intersection(x_ind_set):
                self.axpy(alpha, x.copy(x_ind), ind)
                return

        if ind is None:
            Y = iter(self._list)
            len_Y = len(self._list)
        elif isinstance(ind, Number):
            Y = iter([self._list[ind]])
            len_Y = 1
        else:
            Y = (self._list[i] for i in ind)
            len_Y = len(ind)

        if x_ind is None:
            X = iter(x._list)
            len_X = len(x._list)
        elif isinstance(x_ind, Number):
            X = iter([x._list[x_ind]])
            len_X = 1
        else:
            X = (x._list[i] for i in x_ind)
            len_X = len(x_ind)

        if alpha == 0:
            return
        elif len_X == 1:
            xx = next(X)
            for y in Y:
                y.axpy(alpha, xx)
        else:
            assert len_X == len_Y
            for xx, y in izip(X, Y):
                y.axpy(alpha, xx)

    def dot(self, other, pairwise, ind=None, o_ind=None):
        assert self.check_ind(ind)
        assert other.check_ind(o_ind)
        assert self.dim == other.dim

        if ind is None:
            A = self._list
            len_A = len(A)
        elif isinstance(ind, Number):
            A = [self._list[ind]]
            len_A = 1
        else:
            A = (self._list[i] for i in ind)
            len_A = len(ind)

        if o_ind is None:
            B = other._list
            len_B = len(B)
        elif isinstance(o_ind, Number):
            B = [other._list[o_ind]]
            len_B = 1
        else:
            B = (other._list[i] for i in o_ind)
            len_B = len(o_ind)

        if pairwise:
            assert len_A == len_B
            return np.array([a.dot(b) for a, b in izip(A, B)])
        else:
            A = list(A)
            B = list(B)
            R = np.empty((len_A, len_B))
            for i, a in enumerate(A):
                for j, b in enumerate(B):
                    R[i, j] = a.dot(b)
            return R

    def gramian(self, ind=None):
        assert self.check_ind(ind)

        if ind is None:
            A = self._list
        elif isinstance(ind, Number):
            A = [self._list[ind]]
        else:
            A = [self._list[i] for i in ind]

        R = np.empty((len(A), len(A)))
        for i in xrange(len(A)):
            for j in xrange(i, len(A)):
                R[i, j] = A[i].dot(A[j])
                R[j, i] = R[i, j]
        return R

    def lincomb(self, coefficients, ind=None):
        assert self.check_ind(ind)
        assert 1 <= coefficients.ndim <= 2

        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, :]

        if ind is None:
            V = self._list
        elif isinstance(ind, Number):
            V = [self._list[ind]]
        else:
            V = [self._list[i] for i in ind]

        assert coefficients.shape[1] == self.len_ind(ind)

        RL = []
        for coeffs in coefficients:
            R = self.vector_type.zeros(self.dim)
            for v, c in izip(V, coeffs):
                R.axpy(c, v)
            RL.append(R)

        return type(self)(RL, dim=self.dim, copy=False)

    def l1_norm(self, ind=None):
        assert self.check_ind(ind)

        if ind is None:
            ind = xrange(len(self._list))
        elif isinstance(ind, Number):
            ind = [ind]

        return np.array([self._list[i].l1_norm() for i in ind])

    def l2_norm(self, ind=None):
        assert self.check_ind(ind)

        if ind is None:
            ind = xrange(len(self._list))
        elif isinstance(ind, Number):
            ind = [ind]

        return np.array([self._list[i].l2_norm() for i in ind])

    def components(self, component_indices, ind=None):
        assert self.check_ind(ind)
        assert isinstance(component_indices, list) and (len(component_indices) == 0 or min(component_indices) >= 0) \
            or (isinstance(component_indices, np.ndarray) and component_indices.ndim == 1
                and (len(component_indices) == 0 or np.min(component_indices) >= 0))

        if ind is None:
            ind = xrange(len(self._list))
        elif isinstance(ind, Number):
            ind = [ind]

        if len(ind) == 0:
            assert len(component_indices) == 0 \
                or isinstance(component_indices, list) and max(component_indices) < self.dim \
                or isinstance(component_indices, np.ndarray) and np.max(component_indices) < self.dim
            return np.empty((0, len(component_indices)))

        R = np.empty((len(ind), len(component_indices)))
        for k, i in enumerate(ind):
            R[k] = self._list[i].components(component_indices)

        return R

    def amax(self, ind=None):
        assert self.check_ind(ind)
        assert self.dim > 0

        if ind is None:
            ind = xrange(len(self._list))
        elif isinstance(ind, Number):
            ind = [ind]

        MI = np.empty(len(ind), dtype=np.int)
        MV = np.empty(len(ind))

        for k, i in enumerate(ind):
            MI[k], MV[k] = self._list[i].amax()

        return MI, MV

    def __str__(self):
        return 'ListVectorArray of {} {}s of dimension {}'.format(len(self._list), str(self.vector_type), self._dim)


class NumpyListVectorArray(ListVectorArray):
    vector_type = NumpyVector
