# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
from itertools import izip

import pymor.core as core
from pymor.la import NumpyVectorArray
from pymor.playground.la import BlockVectorArray
from pymor.reductors.basic import GenericRBReconstructor


class GenericBlockRBReconstructor(core.BasicInterface):

    def __init__(self, RB):
        assert isinstance(RB, list)
        self.RB = RB

    def reconstruct(self, U):
        """Reconstruct high-dimensional vector from reduced vector `U`."""
        assert isinstance(U, BlockVectorArray)
        assert all(subspace.type == NumpyVectorArray for subspace in U.space.subtype)
        return BlockVectorArray([GenericRBReconstructor(rb).reconstruct(block)
                                 for rb, block in izip(self.RB, U._blocks)])

    def restricted_to_subbasis(self, dim):
        raise NotImplementedError
        """Analog of :meth:`~pymor.operators.basic.NumpyMatrixOperator.projected_to_subbasis`."""
        assert dim <= len(self.RB)
        return GenericBlockRBReconstructor([rb.copy(ind=range(dim)) for rb in self.RB])

