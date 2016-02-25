# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.basic import OperatorBase
from pymor.vectorarrays.interfaces import VectorSpace
from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorArray

import numpy as np
import math as m
from discretization import Vector, DiffusionOperator


class WrappedVector(CopyOnWriteVector):

    def __init__(self, vector):
        assert isinstance(vector, Vector)
        self._impl = vector

    @classmethod
    def from_instance(cls, instance):
        return cls(instance._impl)

    @classmethod
    def make_zeros(cls, subtype):
        return cls(Vector(subtype, 0))

    @property
    def subtype(self):
        return self._impl.dim

    @property
    def dim(self):
        return self._impl.dim

    @property
    def data(self):
        return np.frombuffer(self._impl.data(), dtype=np.float)

    def _copy_data(self):
        self._impl = Vector(self._impl)

    def _scal(self, alpha):
        self._impl.scal(alpha)

    def _axpy(self, alpha, x):
        self._impl.axpy(alpha, x._impl)

    def dot(self, other):
        return self._impl.dot(other._impl)

    def l1_norm(self):
        raise NotImplementedError

    def l2_norm(self):
        return m.sqrt(self.dot(self))

    def sup_norm(self):
        raise NotImplementedError

    def components(self, component_indices):
        raise NotImplementedError

    def amax(self):
        raise NotImplementedError


class WrappedDiffusionOperator(OperatorBase):
    def __init__(self, op):
        assert isinstance(op, DiffusionOperator)
        self._impl = op
        self.source = VectorSpace(ListVectorArray, (WrappedVector, op.dim_source))
        self.range = VectorSpace(ListVectorArray, (WrappedVector, op.dim_range))
        self.linear = True

    @classmethod
    def create(cls, n, left, right):
        return cls(DiffusionOperator(n, left, right))

    def apply(self, U, ind=None, mu=None):
        assert U in self.source

        if ind is None:
            ind = range(len(U))

        def apply_one_vector(u):
            v = Vector(self.range.dim, 0)
            self._impl.apply(u._impl, v)
            return WrappedVector(v)

        return ListVectorArray([apply_one_vector(U._list[i]) for i in ind], subtype=self.range.subtype)
