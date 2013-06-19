# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number
from itertools import izip
import math as m

import numpy as np

from pymor.la.interfaces import VectorArrayInterface
from pymor.core import defaults
from dunelinearellipticcg2dsgrid import DuneVector

class DuneVectorArray(VectorArrayInterface):

    def empty(cls, dim, reserve=0):
        return cls([], dim=dim)

    def __init__(self, vectors, dim=None):
        if isinstance(vectors, DuneVector):
            self._vectors = [vectors]
            self._dim = vectors.len()
            assert dim is None or dim == self._dim
        else:
            self._vectors = list(vectors)
            if len(self._vectors) == 0:
                assert dim is not None
                self._dim = dim
            else:
                self._dim = self._vectors[0].len()
                assert all(v.len() == self._dim for v in self._vectors)

    def __len__(self):
        return len(self._vectors)

    @property
    def dim(self):
        return self._dim

    def copy(self, ind=None):
        if ind is None:
            return DuneVectorArray([DuneVector(v) for v in self._vectors], self.dim)
        else:
            return DuneVectorArray([DuneVector(self._vectors[i]) for i in ind], self.dim)

    def append(self, other, o_ind=None, remove_from_other=False):
        assert other.dim == self.dim
        new_vectors = other._vectors if o_ind is None else [other._vectors[i] for i in o_ind]
        if not remove_from_other:
            self._vectors.extend([DuneVector(v) for v in new_vectors])
        else:
            self._vectors.extend(new_vectors)
            o_ind = o_ind or []
            other._vectors = [v for i, v in enumerate(other._vectors) if i not in o_ind]

    def remove(self, ind):
        ind = ind or []
        self._vectors = [v for i, v in enumerate(self._vectors) if i not in ind]

    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        assert other.dim == self.dim
        if ind == None:
            c = DuneVectorArray.empty(self.dim)
            c.append(other, o_ind=o_ind, remove_from_other=remove_from_other)
            assert len(c) == len(self)
            self._vectors = c._vectors
        else:
            o_ind = o_ind or xrange(len(other))
            assert len(ind) == len(o_ind)
            if not remove_from_other:
                for i, oi in izip(ind, o_ind):
                    self._vectors[i] = DuneVector(other._vectors[oi])
            else:
                for i, oi in izip(ind, o_ind):
                    self._vectors[i] = other._vectors[oi]
                other._vectors = [v for i, v in enumerate(other._vectors) if i not in o_ind]

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        assert self._compatible_shape(other, ind, o_ind)
        rtol = rtol or defaults.float_cmp_tol
        atol = atol or rtol
        A = self._vectors if ind is None else [self._vectors[i] for i in ind]
        B = other._vectors if o_ind is None else [other._vectors[i] for i in o_ind]

        def vec_almost_equal(v, w):
            ws = DuneVector(w)
            ws.scale(-1)
            error = v.add(ws)
            del ws
            error_norm = m.sqrt(error.dot(error))
            w_norm = m.sqrt(w.dot(w))
            return error_norm <= atol + w_norm*rtol

        return np.array([vec_almost_equal(v, w) for v, w in izip(A, B)])

    def add_mult(self, other, factor=1., o_factor=1., ind=None, o_ind=None):
        assert other is not None or o_factor == 0
        if isinstance(other, Number):
            assert other == 0
            other = None
            o_factor = 0
        if other is not None:
            assert self._compatible_shape(other, ind, o_ind)
        if o_factor == 0:
            R = self.copy(ind)
            for v in R._vectors:
                v.scale(factor)
            return R
        else:
            ind = ind or xrange(len(self))
            o_ind = o_ind or xrange(len(other))
            if factor == 1:
                S1 = [self._vectors[i] for i in ind]
            else:
                S1 = [DuneVector(self._vectors[i]) for i in ind]
                for v in S1:
                    v.scale(factor)
            if o_factor == 1:
                S2 = [other._vectors[i] for i in o_ind]
            else:
                S2 = [DuneVector(other._vectors[i]) for i in o_ind]
                for v in S2:
                    v.scale(o_factor)
        return DuneVectorArray([v1.add(v2) for v1, v2 in izip(S1, S2)], dim=self._dim)

    def iadd_mult(self, other, factor=1., o_factor=1., ind=None, o_ind=None):
        assert other is None or o_factor != 0
        if other is not None:
            assert self._compatible_shape(other, ind, o_ind)
        if o_factor == 0:
            ind = ind or xrange(len(self))
            for i in ind:
                self._vectors[i].scale(factor)
        else:
            self.replace(self.add_mult(other, factor=factor, o_factor=o_factor, ind=ind, o_ind=o_ind),
                         ind=ind, remove_from_other=True)
        return self

    def prod(self, other, ind=None, o_ind=None, pairwise=True):
        F1 = self._vectors if ind is None else [self._vectors[i] for i in ind]
        F2 = other._vectors if o_ind is None else [other._vectors[i] for i in o_ind]
        if pairwise:
            assert self.dim == other.dim
            assert len(F1) == len(F2)
            return np.array([v1.dot(v2) for v1, v2 in izip(F1, F2)])
        else:
            assert self.dim == other.dim
            R = np.empty((len(F1), len(F2)))
            for i, v1 in enumerate(F1):
                for j, v2 in enumerate(F2):
                    R[i, j] = v1.dot(v2)
            return R

    def lincomb(self, factors, ind=None):
        if factors.ndim > 1:
            if len(factors) > 1:
                raise NotImplementedError
            else:
                factors = factors.ravel()
        V = self.copy(ind)._vectors
        assert len(V) == len(factors)
        if len(V) == 0:
            return DuneVectorArray(DuneVector(self.dim))
        for v, f in izip(V, factors):
            v.scale(f)
        R = V[0]
        for v in V[1:]:
            R = R.add(v)
        return DuneVectorArray(R)

    def lp_norm(self, p, ind=None):
        if p != 2:
            raise NotImplementedError
        return np.sqrt(self.prod(self, ind=ind, o_ind=ind, pairwise=True))
