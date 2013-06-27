# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip, repeat

from pymor.core.interfaces import BasicInterface, abstractmethod, abstractproperty
from pymor.core.exceptions import CommunicationError
from pymor.la.interfaces import VectorArrayInterface


class ListVectorArray(VectorArrayInterface):

    vector_type = None

    def __init__(self, vectors, dim=None, copy=True):
        if not copy and isinstance(vectors, list):
            self._list = vectors
        else:
            self._list = list(vectors)
        if copy:
            self._list = [v.copy() for v in self._list]
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
        return cls([vector_type.zeros(dim) for c in xrange(count)], dim=dim, copy=False)

    def __len__(self):
        return len(self._list)

    @property
    def dim(self):
        return self._dim

    def copy(self, ind=None):
        ind = None if ind is None else np.array(ind, copy=False, ndmin=1, dtype=np.int).ravel()

        if ind is None:
            return type(self).(self._list, dim=self._dim, copy=True)
        else:
            return type(self).([self._list[i] for i in ind], dim=self._dim, copy=True)

    def append(self, other, o_ind=None, remove_from_other=False):
        o_ind = None if o_ind is None else np.array(o_ind, copy=False, ndmin=1, dtype=np.int).ravel()
        assert other.dim == self.dim

        other_list = other._list
        if not remove_from_other:
            if o_ind is None:
                self._list.extend([v.copy() for v in other_list])
            else:
                self._list.extend([other_list[i].copy() for i in o_ind])
        else:
            if o_ind is None:
                self._list.extend(other_list)
                other._list = []
            else:
                self._list.extend([other_list[i] for i in o_ind])
                other._list = [v for i, v in enumerate(other._vectors) if i not in o_ind]

    def remove(self, ind):
        ind = None if ind is None else np.array(ind, copy=False, ndmin=1, dtype=np.int).ravel()
        self._list = [] if ind is None else [v for i, v in enumerate(self._list) if i not in ind]

    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        ind = None if ind is None else np.array(ind, copy=False, ndmin=1, dtype=np.int).ravel()
        o_ind = None if o_ind is None else np.array(o_ind, copy=False, ndmin=1, dtype=np.int).ravel()
        assert other.dim == self.dim

        if ind == None:
            c = type(self).empty(self.dim)
            c.append(other, o_ind=o_ind, remove_from_other=remove_from_other)
            assert len(c) == len(self)
            self._list = c._list
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
        ind = None if ind is None else np.array(ind, copy=False, ndmin=1, dtype=np.int).ravel()
        o_ind = None if o_ind is None else np.array(o_ind, copy=False, ndmin=1, dtype=np.int).ravel()
        assert other.dim == self.dim

        if ind is None:
            A = self._list
            len_A = len(A)
        else:
            A = (self._list[i] for i in ind)
            len_A = len(ind)

        if o_ind is None:
            B = other._list
            len_B = len(B)
        else:
            B = (other._list[i] for i in o_ind)
            len_B = len(o_ind)

        if len_A == 1:
            v = next(A)
            return np.array([v.almost_equal(w) for w in B])
        elif len_B == 1:
            w = next(B)
            return np.array([v.almost_equal(w) for v in A])
        else:
            assert len_A == len_B
            return np.array([v.almost_equal(w) for v, w in izip(A, B)])


    def scal(self, alpha, ind=None):
        ind = None if ind is None else np.array(ind, copy=False, ndmin=1, dtype=np.int).ravel()

        if ind is None:
            for v in self._list:
                v.scal(alpha)
        else:
            l = self._list
            for i in ind:
                l[i].scal(alpha)

    def axpy(self, alpha, x, ind=None, x_ind=None):
        ind = None if ind is None else np.array(ind, copy=False, ndmin=1, dtype=np.int).ravel()
        x_ind = None if x_ind is None else np.array(x_ind, copy=False, ndmin=1, dtype=np.int).ravel()

        if ind is None:
            Y = self._list
            len_Y = len(Y)
        else:
            Y = (self._list[i] for i in ind)
            len_Y = len(ind)

        if x_ind is None:
            X = x._list
            len_X = len(X)
        else:
            X = (x._list[i] for i in x_ind)
            len_X = len(x_ind)

        if alpha == 0:
            for y in Y:
                y.set_zero()
        elif alpha == 1:
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
        ind = None if ind is None else np.array(ind, copy=False, ndmin=1, dtype=np.int).ravel()
        o_ind = None if o_ind is None else np.array(o_ind, copy=False, ndmin=1, dtype=np.int).ravel()
        assert self.dim == other.dim

        if ind is None:
            A = self._list
            len_A = len(A)
        else:
            A = (self._list[i] for i in ind)
            len_A = len(ind)

        if o_ind is None:
            B = other._list
            len_B = len(B)
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
        ind = None if ind is None else np.array(ind, copy=False, ndmin=1, dtype=np.int).ravel()

        A = self._list if ind is None else [self._list[i] for i in ind]

        R = np.empty((len(A), len(A)))
        for i in xrange(len(A)):
            for j in xrange(i, len(A)):
                R[i, j] = A[i].dot(A[j])
                R[j, i] = R[i, j]
        return R
