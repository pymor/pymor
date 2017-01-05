# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

if config.HAVE_FENICS:
    import dolfin as df
    from numbers import Number

    from pymor.core.defaults import defaults
    from pymor.operators.basic import OperatorBase
    from pymor.operators.constructions import ZeroOperator
    from pymor.vectorarrays.fenics import FenicsVectorSpace


    class FenicsMatrixOperator(OperatorBase):
        """Wraps a FEniCS matrix as an |Operator|."""

        linear = True

        def __init__(self, matrix, source_space, range_space, solver_options=None, name=None):
            assert matrix.rank() == 2
            self.source = FenicsVectorSpace(source_space)
            self.range = FenicsVectorSpace(range_space)
            self.matrix = matrix
            self.solver_options = solver_options
            self.name = name

        def apply(self, U, mu=None):
            assert U in self.source
            R = self.range.zeros(len(U))
            for u, r in zip(U._list, R._list):
                self.matrix.mult(u.impl, r.impl)
            return R

        def apply_transpose(self, V, mu=None):
            assert V in self.range
            U = self.source.zeros(len(V))
            for v, u in zip(V._list, U._list):
                self.matrix.transpmult(v.impl, u.impl)
            return U

        def apply_inverse(self, V, mu=None, least_squares=False):
            assert V in self.range
            if least_squares:
                raise NotImplementedError
            R = self.source.zeros(len(V))
            options = self.solver_options.get('inverse') if self.solver_options else None
            for r, v in zip(R._list, V._list):
                _apply_inverse(self.matrix, r.impl, v.impl, options)
            return R

        def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
            if not all(isinstance(op, (FenicsMatrixOperator, ZeroOperator)) for op in operators):
                return None
            assert not solver_options

            if coefficients[0] == 1:
                matrix = operators[0].matrix.copy()
            else:
                matrix = operators[0].matrix * coefficients[0]
            for op, c in zip(operators[1:], coefficients[1:]):
                if isinstance(op, ZeroOperator):
                    continue
                matrix.axpy(c, op.matrix, False)  # in general, we cannot assume the same nonzero pattern for
                                                  # all matrices. how to improve this?

            return FenicsMatrixOperator(matrix, self.source.V, self.range.V, name=name)


    @defaults('solver', 'preconditioner')
    def _solver_options(solver='bicgstab', preconditioner='amg'):
        return {'solver': solver, 'preconditioner': preconditioner}


    def _apply_inverse(matrix, r, v, options=None):
        options = options or _solver_options()
        solver = options.get('solver')
        preconditioner = options.get('preconditioner')
        # preconditioner argument may only be specified for iterative solvers:
        options = (solver, preconditioner) if preconditioner else (solver,)
        df.solve(matrix, r, v, *options)
