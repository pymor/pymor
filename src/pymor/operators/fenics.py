# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

try:
    import dolfin as df
    HAVE_FENICS = True
except ImportError:
    HAVE_FENICS = False

if HAVE_FENICS:
    from itertools import izip
    from numbers import Number

    from pymor.operators.basic import OperatorBase
    from pymor.operators.constructions import ZeroOperator
    from pymor.vectorarrays.fenics import FenicsVectorSpace


    class FenicsMatrixOperator(OperatorBase):
        """Wraps a FEniCS matrix as an |Operator|."""

        linear = True

        def __init__(self, matrix, name=None):
            assert matrix.rank() == 2
            comm = matrix.mpi_comm()
            self.source = FenicsVectorSpace(matrix.size(1), mpi_comm=comm)
            self.range = FenicsVectorSpace(matrix.size(0), mpi_comm=comm)
            self.name = name
            self.matrix = matrix

        def apply(self, U, ind=None, mu=None):
            assert U in self.source
            assert U.check_ind(ind)
            vectors = U._list if ind is None else [U._list[ind]] if isinstance(ind, Number) else [U._list[i] for i in ind]
            R = self.range.zeros(len(vectors))
            for u, r in zip(vectors, R._list):
                self.matrix.mult(u.impl, r.impl)
            return R

        def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
            assert U in self.range
            assert U.check_ind(ind)
            assert source_product is None or source_product.source == source_product.range == self.source
            assert range_product is None or range_product.source == range_product.range == self.range
            if range_product:
                PrU = range_product.apply(U, ind=ind)._list
            else:
                PrU = U._list if ind is None else [U._list[ind]] if isinstance(ind, Number) else [U._list[i] for i in ind]
            ATPrU = self.source.zeros(len(PrU))
            for u, r in zip(PrU, ATPrU._list):
                self.matrix.transpmult(u.impl, r.impl)
            if source_product:
                return source_product.apply_inverse(ATPrU)
            else:
                return ATPrU

        def apply_inverse(self, V, ind=None, mu=None, least_squares=False):
            assert V in self.range
            if least_squares:
                raise NotImplementedError
            vectors = V._list if ind is None else [V._list[ind]] if isinstance(ind, Number) else [V._list[i] for i in ind]
            R = self.source.zeros(len(vectors))
            for r, v in zip(R._list, vectors):
                df.solve(self.matrix, r.impl, v.impl)
            return R

        def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
            if not all(isinstance(op, (FenicsMatrixOperator, ZeroOperator)) for op in operators):
                return None
            assert not solver_options

            if coefficients[0] == 1:
                matrix = operators[0].matrix.copy()
            else:
                matrix = operators[0].matrix * coefficients[0]
            for op, c in izip(operators[1:], coefficients[1:]):
                if isinstance(op, ZeroOperator):
                    continue
                matrix.axpy(c, op.matrix, False)  # in general, we cannot assume the same nonzero pattern for
                                                  # all matrices. how to improve this?

            return FenicsMatrixOperator(matrix, name=name)
