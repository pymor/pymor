# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np

from pymor.la import NumpyVectorArray
from pymor.playground.la.dunevectorarray import DuneVectorArray
from pymor.operators.interfaces import LinearOperatorInterface

from dunelinearellipticcg2dsgrid import DuneVector


class DuneLinearOperator(LinearOperatorInterface):

    type_source = type_range = DuneVectorArray

    def __init__(self, dune_op, dim, name=None):
        super(DuneLinearOperator, self).__init__()
        self.dim_source = dim
        self.dim_range = dim
        self.name = name
        self.dune_op = dune_op

    def _assemble(self, mu=None):
        assert mu is None
        return self

    def assemble(self, mu=None, force=False):
        assert mu is None
        return self

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, DuneVectorArray)
        assert mu is None
        vectors = U._vectors if ind is None else [U._vectors[i] for i in ind]
        return DuneVectorArray([self.dune_op.apply(v) for v in vectors], dim=self.dim_source)


class DuneLinearFunctional(LinearOperatorInterface):

    type_source = DuneVectorArray
    type_range = NumpyVectorArray

    def __init__(self, dune_vec, name=None):
        super(DuneLinearFunctional, self).__init__()
        self.dim_source = dune_vec.len()
        self.dim_range = 1
        self.name = name
        self.dune_vec = dune_vec

    def _assemble(self, mu=None):
        assert mu is None
        return self

    def assemble(self, mu=None, force=False):
        assert mu is None
        return self

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, DuneVectorArray)
        assert mu is None
        vectors = U._vectors if ind is None else [U._vectors[i] for i in ind]
        if len(vectors) == 0:
            return NumpyVectorArray.empty(dim=1)
        else:
            return NumpyVectorArray([[self.dune_vec.dot(v)] for v in vectors])

    def as_vector_array(self):
        return DuneVectorArray(DuneVector(self.dune_vec))
