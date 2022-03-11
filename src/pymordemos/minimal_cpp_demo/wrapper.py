# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.interface import Operator
from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace

import numpy as np
import math
from model import Vector, DiffusionOperator


class WrappedVector(CopyOnWriteVector):

    def __init__(self, vector):
        assert isinstance(vector, Vector)
        self._impl = vector

    @classmethod
    def from_instance(cls, instance):
        return cls(instance._impl)

    def to_numpy(self, ensure_copy=False):
        result = np.frombuffer(self._impl, dtype=np.float64)
        if ensure_copy:
            result = result.copy()
        return result

    def _copy_data(self):
        self._impl = Vector(self._impl)

    def _scal(self, alpha):
        self._impl.scal(alpha)

    def _axpy(self, alpha, x):
        self._impl.axpy(alpha, x._impl)

    def inner(self, other):
        return self._impl.inner(other._impl)

    def norm(self):
        return math.sqrt(self.inner(self))

    def norm2(self):
        return self.inner(self)

    def sup_norm(self):
        raise NotImplementedError

    def dofs(self, dof_indices):
        raise NotImplementedError

    def amax(self):
        raise NotImplementedError


class WrappedVectorSpace(ListVectorSpace):

    def __init__(self, dim):
        self.dim = dim

    def zero_vector(self):
        return WrappedVector(Vector(self.dim, 0))

    def make_vector(self, obj):
        return WrappedVector(obj)

    def __eq__(self, other):
        return type(other) is WrappedVectorSpace and self.dim == other.dim


class WrappedDiffusionOperator(Operator):
    def __init__(self, op):
        assert isinstance(op, DiffusionOperator)
        self.op = op
        self.source = WrappedVectorSpace(op.dim_source)
        self.range = WrappedVectorSpace(op.dim_range)
        self.linear = True

    @classmethod
    def create(cls, n, left, right):
        return cls(DiffusionOperator(n, left, right))

    def apply(self, U, mu=None):
        assert U in self.source

        def apply_one_vector(u):
            v = Vector(self.range.dim, 0)
            self.op.apply(u._impl, v)
            return v

        return self.range.make_array([apply_one_vector(u) for u in U.vectors])
