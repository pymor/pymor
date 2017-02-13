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

        def apply(self, U, mu=None):
            assert U in self.source
            R = self.range.zeros(len(U))
            for u, r in zip(U._list, R._list):
                self.matrix.Mult(u.impl, r.impl, 1.)
            return R

        def apply_transpose(self, V, mu=None):
            assert V in self.range
            U = self.source.zeros(len(V))
            mat = self.matrix.Transpose()
            for v, u in zip(V._list, U._list):
                mat.Mult(v.impl, u.impl, 1.)
            return U

        def apply_inverse(self, V, mu=None, least_squares=False):
            assert V in self.range
            if least_squares:
                raise NotImplementedError
            R = self.source.zeros(len(V))
            with ngsolve.TaskManager():
                inv = self.matrix.Inverse(self.free_dofs)
                for r, v in zip(R._list, V._list):
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
