# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_DUNEXT:
    import numpy as np

    from pymor.vectorarrays.list import VectorInterface, ListVectorSpace


    class DuneXTVector(VectorInterface):
        """Wraps a FEniCS vector to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl

        @property
        def data(self):
            return np.frombuffer(self.impl)

        def copy(self, deep=False):
            return DuneXTVector(self.impl.copy())

        def scal(self, alpha):
            self.impl.scal(alpha)

        def axpy(self, alpha, x):
            self.impl.axpy(alpha, x.impl)

        def dot(self, other):
            return self.impl.dot(other.impl)

        def l1_norm(self):
            return self.impl.l1_norm()

        def l2_norm(self):
            return self.impl.l2_norm()

        def l2_norm2(self):
            return self.impl.l2_norm() ** 2

        def sup_norm(self):
            return self.impl.sup_norm()

        def components(self, component_indices):
            impl = self.impl
            return np.array([impl[i] for i in component_indices])

        def amax(self):
            raise NotImplementedError

        def __add__(self, other):
            return DuneXTVector(self.impl + other.impl)

        def __iadd__(self, other):
            self.impl += other.impl
            return self

        __radd__ = __add__

        def __sub__(self, other):
            return DuneXTVector(self.impl - other.impl)

        def __isub__(self, other):
            self.impl -= other.impl
            return self

        def __mul__(self, other):
            return DuneXTVector(self.impl * other)

        def __imul__(self, other):
            self.impl *= other
            return self

        def __neg__(self):
            return self * (-1)



    class DuneXTVectorSpace(ListVectorSpace):

        def __init__(self, vector_type, dim, id_='STATE'):
            self.vector_type = vector_type
            self.dim = dim
            self.id = id_

        def __eq__(self, other):
            return type(other) is DuneXTVectorSpace and self.vector_type == other.vector_type and self.dim == other.dim

        # since we implement __eq__, we also need to implement __hash__
        def __hash__(self):
            return id(self.vector_type) + hash(self.dim)

        def zero_vector(self):
            return DuneXTVector(self.vector_type(self.dim, 0.))

        def make_vector(self, obj):
            return DuneXTVector(obj)

        def vector_from_data(self, data):
            v = self.zero_vector()
            v.data[:] = data
            return v

        @classmethod
        def space_from_vector_obj(cls, vec, id_):
            return cls(type(vec), len(vec), id_)
