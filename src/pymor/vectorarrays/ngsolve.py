# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

if config.HAVE_NGSOLVE:
    from ngsolve import BaseVector
    import numpy as np

    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace


    class NGSolveVector(CopyOnWriteVector):
        """Wraps a NGSolve BaseVector to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl
            self._array = impl.FV().NumPy()

        @classmethod
        def from_instance(cls, instance):
            return cls(instance.impl)

        def _copy_data(self):
            new_impl = BaseVector(self.impl.size)
            new_impl.data = self.impl
            self.impl = new_impl
            self._array = new_impl.FV().NumPy()

        @property
        def data(self):
            return self._array

        def _scal(self, alpha):
            self.impl.data = float(alpha) * self.impl

        def _axpy(self, alpha, x):
            self.impl.data = self.impl + float(alpha) * x.impl

        def dot(self, other):
            return self.impl.InnerProduct(other.impl)

        def l1_norm(self):
            return np.linalg.norm(self._array, ord=1)

        def l2_norm(self):
            return self.impl.Norm()

        def l2_norm2(self):
            return self.impl.Norm() ** 2

        def components(self, component_indices):
            return self._array[component_indices]

        def amax(self):
            A = np.abs(self._array)
            max_ind = np.argmax(A)
            max_val = A[max_ind]
            return max_ind, max_val


    class NGSolveVectorSpace(ListVectorSpace):

        def __init__(self, dim, id_='STATE'):
            self.dim = dim
            self.id = id_

        def __eq__(self, other):
            return type(other) is NGSolveVectorSpace and self.dim == other.dim and self.id == other.id

        def __hash__(self):
            return hash(self.dim) + hash(self.id)

        def zero_vector(self):
            impl = BaseVector(self.dim)
            impl.FV().NumPy()[:] = 0
            return NGSolveVector(impl)

        def make_vector(self, obj):
            return NGSolveVector(obj)
