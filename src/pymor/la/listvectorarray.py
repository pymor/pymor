# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip, repeat
from numbers import Number

import numpy as np

from pymor.core.interfaces import BasicInterface, abstractmethod, abstractclassmethod, abstractproperty
from pymor.core.exceptions import CommunicationError
from pymor.la.interfaces import VectorArrayInterface


class VectorInterface(BasicInterface):
    '''Interface for vectors.

    This Interface is mainly inteded to be used in conjunction with ListVectorArray. In general, all
    pyMor ojects operate on VectorArrays instead of single vectors! All methods of the interface have
    a direct counterpart in VectorArrayInterface.
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


class ListVectorArray(VectorArrayInterface):

    vector_type = None

    def __init__(self, vectors, dim=None, copy=True):
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
        return cls([], dim, copy=False)

    @classmethod
    def zeros(cls, dim, count=1):
        return cls([cls.vector_type.zeros(dim) for c in xrange(count)], dim=dim, copy=False)

    def __len__(self):
        return len(self._list)

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
        assert self.check_ind(o_ind)
        assert other.dim == self.dim

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
                other._list = [v for i, v in enumerate(other._vectors) if i not in o_ind]

    def remove(self, ind):
        assert self.check_ind(ind)
        if ind is None:
            self._list = []
        elif isinstance(ind, Number):
            del self._list[ind]
        else:
            self._list = [v for i, v in enumerate(self._list) if i not in ind]

    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        assert self.check_ind(ind)
        assert self.check_ind(o_ind)
        assert other.dim == self.dim

        if ind == None:
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
                    for i, oi in izip(ind, o_ind):
                        self._list[i] = other._list[oi].copy()
                else:
                    for i, oi in izip(ind, o_ind):
                        self._list[i] = other._list[oi]
                    other._list = [v for i, v in enumerate(other._list) if i not in o_ind]

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        assert self.check_ind(ind)
        assert self.check_ind(o_ind)
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
        assert self.check_ind(ind)

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
        assert self.check_ind(ind)
        assert self.check_ind(x_ind)

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

        if alpha == 1:
            if len_X == 1:
                x = next(X)
                for y in Y:
                    y += x
            else:
                assert len_X == len_Y
                for x, y in izip(X, Y):
                    y += x
        elif alpha == -1:
            if len_X == 1:
                x = next(X)
                for y in Y:
                    y -= x
            else:
                assert len_X == len_Y
                for x, y in izip(X, Y):
                    y -= x
        else:
            if len_X == 1:
                x = next(X) * alpha
                for y in Y:
                    y += x
            else:
                assert len_X == len_Y
                for x, y in izip(X, Y):
                    y += x * alpha

    def dot(self, other, pairwise, ind=None, o_ind=None):
        assert self.check_ind(ind)
        assert self.check_ind(o_ind)
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

        assert coefficients.shape[1] == len(self._list)

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

    def sup_norm(self, ind=None):
        assert self.check_ind(ind)

        if ind is None:
            ind = xrange(len(self._list))
        elif isinstance(ind, Number):
            ind = [ind]

        return np.array([self._list[i].sup_norm() for i in ind])

    def components(self, component_indices, ind=None):
        assert self.check_ind(ind)

        if ind is None:
            ind = xrange(len(self._list))
        elif isinstance(ind, Number):
            ind = [ind]

        R = np.empty((len(ind), len(component_indices)))
        for k, i in enumerate(ind):
            R[k] = self._list[i].components(component_indices)

        return R

    def amax(self, ind=None):
        assert self.check_ind(ind)

        if ind is None:
            ind = xrange(len(self._list))
        elif isinstance(ind, Number):
            ind = [ind]

        MI = np.empty(len(ind))
        MV = np.empty(len(ind))

        for k, i in enumerate(ind):
            MI[k], MV[k] = self._list[i].amax()

        return MI, MV

    def __str__(self):
        return 'ListVectorArray of {} {}s of dimension {}'.format(len(self._list), str(self.vector_type), self._dim)
