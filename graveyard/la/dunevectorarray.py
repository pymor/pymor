# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number
from itertools import izip
import math as m

import numpy as np

from pymor import defaults
from pymor.la.listvectorarray import VectorInterface, ListVectorArray
from dunelinearellipticcg2dsgrid import DuneVector


class WrappedDuneVector(VectorInterface):

    def __init__(self, vector):
        assert isinstance(vector, DuneVector)
        self._vector = vector

    @classmethod
    def zeros(cls, dim):
        return cls(DuneVector(dim))

    @property
    def dim(self):
        return self._vector.len()

    def copy(self):
        return type(self)(DuneVector(self._vector))

    def almost_equal(self, other, rtol=None, atol=None):
        rtol = rtol or defaults.float_cmp_tol
        atol = atol or rtol
        return (self - other).l2_norm() <= atol + other.l2_norm()*rtol

    def scal(self, alpha):
        self._vector.scale(alpha)

    def axpy(self, alpha, x):
        xx = x.copy()
        xx.scal(alpha)
        self._vector = self._vector.add(xx._vector)

    def dot(self, other):
        return self._vector.dot(other._vector)

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


class DuneVectorArray(ListVectorArray):

    vector_type = WrappedDuneVector
