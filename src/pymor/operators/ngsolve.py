# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

try:
    import ngsolve
    HAVE_NGSOLVE = True
except ImportError:
    HAVE_NGSOLVE = False

if HAVE_NGSOLVE:
    from numbers import Number

    from pymor.operators.basic import OperatorBase
    from pymor.operators.constructions import ZeroOperator
    from pymor.vectorarrays.ngsolve import NGSolveVectorSpace


    class NGSolveMatrixOperator(OperatorBase):
        """Wraps a NGSolve matrix as an |Operator|."""

        linear = True

        def __init__(self, matrix, free_dofs=None, name=None):
            self.source = NGSolveVectorSpace(matrix.width)
            self.range = NGSolveVectorSpace(matrix.height)
            self.matrix = matrix
            self.free_dofs = free_dofs
            self.name = name

        def apply(self, U, ind=None, mu=None):
            assert U in self.source
            assert U.check_ind(ind)
            vectors = U._list if ind is None else [U._list[ind]] if isinstance(ind, Number) else [U._list[i] for i in ind]
            R = self.range.zeros(len(vectors))
            for u, r in zip(vectors, R._list):
                self.matrix.Mult(u.impl, r.impl, 1.)
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
            mat = self.matrix.Transpose()
            for u, r in zip(PrU, ATPrU._list):
                mat.Mult(u.impl, r.impl, 1.)
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
            with ngsolve.TaskManager():
                inv = self.matrix.Inverse(self.free_dofs)
                for r, v in zip(R._list, vectors):
                    r.impl.data = inv * v.impl
            return R

        def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
            if not all(isinstance(op, (NGSolveMatrixOperator, ZeroOperator)) for op in operators):
                return None
            assert not solver_options

            matrix = operators[0].matrix.CreateMatrix()
            matrix.AsVector().data = float(coefficients[0]) * matrix.AsVector()
            for op, c in zip(operators[1:], coefficients[1:]):
                if isinstance(op, ZeroOperator):
                    continue
                matrix.AsVector().data += float(c) * op.matrix.AsVector()
            return NGSolveMatrixOperator(matrix, self.free_dofs, name=name)
