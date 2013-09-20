# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np

from pymor.la import NumpyVectorArray
from pymor.playground.la.dunevectorarray import DuneVectorArray, WrappedDuneVector
from pymor.operators import OperatorBase

from dunelinearellipticcg2dsgrid import DuneVector


class DuneLinearOperator(OperatorBase):

    type_source = type_range = DuneVectorArray
    assembled = True
    sparse = True
    linear = True

    def __init__(self, dune_op, dim, name=None):
        super(DuneLinearOperator, self).__init__()
        self.dim_source = dim
        self.dim_range = dim
        self.name = name
        self.dune_op = dune_op

    def assemble(self, mu=None):
        assert self.check_parameter(mu)
        return self

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, DuneVectorArray)
        assert self.check_parameter(mu)
        vectors = U._list if ind is None else [U._list[i] for i in ind]
        return DuneVectorArray([WrappedDuneVector(self.dune_op.apply(v._vector)) for v in vectors], dim=self.dim_source)


class DuneLinearFunctional(OperatorBase):

    type_source = DuneVectorArray
    type_range = NumpyVectorArray
    linear = True
    assembled = True
    sparse = False

    def __init__(self, dune_vec, name=None):
        super(DuneLinearFunctional, self).__init__()
        self.dim_source = dune_vec.len()
        self.dim_range = 1
        self.name = name
        self.dune_vec = dune_vec

    def assemble(self, mu=None):
        assert self.check_parameter(mu)
        return self

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, DuneVectorArray)
        assert self.check_parameter(mu)
        vectors = U._list if ind is None else [U._list[i] for i in ind]
        if len(vectors) == 0:
            return NumpyVectorArray.empty(dim=1)
        else:
            return NumpyVectorArray([[self.dune_vec.dot(v._vector)] for v in vectors])

    def as_vector(self):
        return DuneVectorArray([WrappedDuneVector(DuneVector(self.dune_vec))])
