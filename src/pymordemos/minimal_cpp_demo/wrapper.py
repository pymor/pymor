from __future__ import absolute_import, division, print_function

from pymor.la.listvectorarray import VectorInterface, ListVectorArray
from pymor import defaults
from pymor.operators.basic import OperatorBase

import numpy as np
import math as m
from discretization import Vector, DiffusionOperator


class WrappedVector(VectorInterface):

    def __init__(self, vector):
        assert isinstance(vector, Vector)
        self._impl = vector

    @classmethod
    def zeros(cls, dim):
        return cls(Vector(dim, 0))

    # naming is consistent with numpy.full in numpy >= 1.8.0
    @classmethod
    def full(cls, dim, value):
        return cls(Vector(dim, value))

    @property
    def dim(self):
        return self._impl.dim

    @property
    def data(self):
        return np.frombuffer(self._impl.data(), dtype=np.float)

    def copy(self):
        return type(self)(Vector(self._impl))

    def almost_equal(self, other, rtol=None, atol=None):
        rtol = rtol if rtol is not None else defaults.float_cmp_tol
        atol = atol or rtol
        return self._impl.almost_equal(other._impl, rtol, atol)

    def scal(self, alpha):
        self._impl.scal(alpha)

    def axpy(self, alpha, x):
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


class WrappedVectorArray(ListVectorArray):
    vector_type = WrappedVector


class WrappedDiffusionOperator(OperatorBase):
    def __init__(self, op):
        assert isinstance(op, DiffusionOperator)
        self._impl = op
        self.dim_source = op.dim_source
        self.dim_range = op.dim_range
        self.type_range = self.type_source = WrappedVectorArray
        self.linear = True

    @classmethod
    def create(cls, n, left, right):
        return cls(DiffusionOperator(n, left, right))

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)

        if ind is None:
            ind = xrange(len(U))

        def apply_one_vector(u):
            v = Vector(self.dim_range, 0)
            self._impl.apply(u._impl, v)
            return WrappedVector(v)

        return WrappedVectorArray([apply_one_vector(U._list[i]) for i in ind], dim=self.dim_range)
