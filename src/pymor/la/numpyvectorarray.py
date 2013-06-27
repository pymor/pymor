# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
from scipy.sparse import issparse

from pymor.core.interfaces import BasicInterface, abstractmethod, abstractproperty
from pymor.core.exceptions import CommunicationError
from pymor.la.interfaces import VectorArrayInterface, Communicable
from pymor.tools import float_cmp


class NumpyVectorArray(VectorArrayInterface, Communicable):

    @classmethod
    def empty(cls, dim, reserve=0):
        va = cls(np.empty((0, 0)))
        va._array = np.empty((reserve, dim))
        va._len = 0
        return va

    @classmethod
    def zeros(cls, dim, count=1):
        return cls(np.zeros((count, dim)))

    def __init__(self, object, dtype=None, copy=False, order=None, subok=False):
        if isinstance(object, np.ndarray) and not copy:
            self._array = object
        elif isinstance(object, Communicable):
            self._array = object.data
            if copy:
                self._array = self._array.copy()
        elif issparse(object):
            self._array = np.array(object.todense(), copy=False)
        else:
            self._array = np.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=2)
        if self._array.ndim != 2:
            assert self._array.ndim == 1
            self._array = np.reshape(self._array, (1,-1))
        self._len = len(self._array)

    def _data(self):
        return self._array[:self._len]

    def __len__(self):
        return self._len

    @property
    def dim(self):
        return self._array.shape[1]

    def copy(self, ind=None):
        if ind is None:
            return NumpyVectorArray(self._array[:self._len], copy=True)
        else:
            C = NumpyVectorArray(self._array[ind], copy=False)
            if not C._array.flags['OWNDATA']:
                C._array = np.array(C._array)
            return C

    def append(self, other, o_ind=None, remove_from_other=False):
        if o_ind == None:
            len_other = other._len
            if len_other <= self._array.shape[0] - self._len:
                self._array[self._len:self._len + len_other] = other._array
            else:
                self._array = np.vstack((self._array[:self._len], other._array[:len_other]))
            self._len += len_other
        else:
            len_other = len[o_ind]
            if len_other <= self._array.shape[0] - self._len:
                self._array[self._len:self._len + len_other] = other._array[o_ind]
            else:
                self._array = np.vstack((self._array[:self._len], other._array[o_ind]))
            self._len += len_other
        if remove_from_other:
            if o_ind == None:
                other._array = np.zeros((0, other.dim))
                other._len = 0
            else:
                other._array = other._array[list(x for x in xrange(len(other)) if x not in o_ind)]
                other._len -= len(o_ind)

    def remove(self, ind):
        if ind == None:
            self._array = np.zeros((0, self.dim))
            self._len = 0
        else:
            self._array = self._array[list(x for x in xrange(len(self)) if x not in ind)]
            self._len = self._array.shape[0]
        if not self._array.flags['OWNDATA']:
            self._array = self._array.copy()

    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        assert self._compatible_shape(other, ind, o_ind)
        if ind is None:
            if o_ind is None:
                self._array = other._array[:other._len]
            else:
                self._array = other._array[o_ind]
            self._len = self._array.shape[0]
        else:
            if o_ind is None:
                self._array[ind] = other._array[:other._len]
            else:
                self._array[ind] = other._array[o_ind]
        if not self._array.flags['OWNDATA']:
            self._array = self._array.copy()
        if remove_from_other:
            if o_ind == None:
                other._array = np.zeros((0, other.dim))
                other._len = 0
            else:
                other._array = other._array[list(x for x in xrange(len(other)) if x not in o_ind)]
                other._len = other._array.shape[0]
            if not other._array.flags['OWNDATA']:
                other._array = self._array.copy()

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        assert self._compatible_shape(other, ind, o_ind)
        A = self._array[:self._len] if ind is None else self._array[ind]
        B = other._array[:other._len] if o_ind is None else other._array[o_ind]
        R = np.all(float_cmp(A, B, rtol=rtol, atol=atol), axis=1).squeeze()
        if R.ndim == 0:
            R = R[np.newaxis, ...]
        return R

    def scal(self, alpha, ind=None):
        if ind is None:
            self._array[:self._len] *= alpha
        else:
            self._array[ind] *= alpha

    def axpy(self, alpha, x, ind=None, x_ind=None):
        assert self._compatible_shape(x, ind, x_ind)
        B = x._array[:x._len] if x_ind is None else x._array[x_ind]

        if alpha == 0:
            if ind is None:
                self._array[:self._len] = 0
            else:
                self._array[ind] = 0
        elif alpha == 1:
            if ind is None:
                self._array[:self._len] += B
            else:
                self._array[ind] += B
        elif alpha == -1:
            if ind is None:
                self._array[:self._len] -= B
            else:
                self._array[ind] -= B
        else:
            if ind is None:
                self._array[:self._len] += B * alpha
            else:
                self._array[ind] += B * alpha

    def dot(self, other, pairwise, ind=None, o_ind=None):
        A = self._array[:self._len] if ind is None else self._array[ind]
        B = other._array[:other._len] if o_ind is None else other._array[o_ind]
        if pairwise:
            assert self._compatible_shape(other, ind, o_ind, broadcast=False)
            return np.sum(A * B, axis=1)
        else:
            assert self.dim == other.dim
            return A.dot(B.T)

    def lincomb(self, coefficients, ind=None):
        assert 1 <= coefficients.ndim <= 2
        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, ...]
        if ind is None:
            assert len(self) == coefficients.shape[1]
        else:
            assert len(ind) == coefficients.shape[1]
        return NumpyVectorArray(coefficients.dot(self._array[:self._len]), copy=False)

    def l1_norm(self, p, ind=None):
        A = self._array[:self._len] if ind is None else self._array[ind]
        return np.sum(np.abs(A), axis=1)

    def l2_norm(self, p, ind=None):
        A = self._array[:self._len] if ind is None else self._array[ind]
        return np.sum(np.power(A, 2), axis=1)**(1/2)

    def components(self, component_indices, ind=None):
        A = self._array[:self._len] if ind is None else self._array[ind]
        return A[:, component_indices]

    def amax(self, ind=None):
        A = self._array[:self._len] if ind is None else self._array[ind]
        A = np.abs(A)
        max_ind = np.argmax(A, axis=1)
        max_val = np.max(A, axis=1)
        return (max_ind, max_val)

    def __str__(self):
        return self._array[:self._len].__str__()

    def __repr__(self):
        return 'NumpyVectorArray({})'.format(self._array[:self._len].__str__())

    def _compatible_shape(self, other, ind=None, o_ind=None, broadcast=True):
        if self.dim != other.dim:
            return False
        if broadcast:
            if o_ind == None and len(other) == 1:
                return True
            elif o_ind != None and len(o_ind) == 1:
                return True
        if ind is None:
            if len(self) == 1:
                return True
            if o_ind is None:
                return len(self) == len(other)
            else:
                return len(self) == len(oind)
        else:
            if len(ind) == 1:
                return True
            if o_ind is None:
                return len(ind) == len(other)
            else:
                return len(ind) == len(oind)
