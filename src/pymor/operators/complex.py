# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Module for complex operators."""

from pymor.operators.basic import OperatorBase
from pymor.operators.block import BlockOperator
from pymor.vectorarrays.block import BlockVectorSpace


class ComplexOperator(OperatorBase):
    def __init__(self, real, imag, solver_options=None, name=None):
        assert real.source == imag.source
        assert real.range == imag.range
        self.__auto_init(locals())
        self.source = real.source
        self.range = real.range

    @property
    def H(self):
        options = (
            {'inverse': self.solver_options.get('inverse_adjoint'),
             'inverse_adjoint': self.solver_options.get('inverse')}
            if self.solver_options else None
        )
        return ComplexOperator(self.real, -self.imag,
                               solver_options=options,
                               name=self.name + '_adjoint')

    def apply(self, U, mu=None):
        assert U in self.source
        real = self.real.apply(U.real, mu=mu) - self.imag.apply(U.imag, mu=mu)
        imag = self.real.apply(U.imag, mu=mu) + self.imag.apply(U.real, mu=mu)
        return real + imag * 1j

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        real = (self.real.apply_adjoint(V.real, mu=mu)
                + self.imag.apply_adjoint(V.imag, mu=mu))
        imag = (self.real.apply_adjoint(V.imag, mu=mu)
                - self.imag.apply_adjoint(V.real, mu=mu))
        return real + imag * 1j

    def apply_inverse(self, V, mu=None, least_squares=False):
        op = BlockOperator([[self.real, -self.imag], [self.imag, self.real]])
        rhs = BlockVectorSpace([self.range, self.range]).make_array([V.real, V.imag])
        U = op.apply_inverse(rhs)
        return U.block(0) + U.block(1) * 1j

    def apply_inverse_adjoint(self, U, mu=None, least_squares=False):
        op = BlockOperator([[self.real, -self.imag], [self.imag, self.real]])
        rhs = BlockVectorSpace([self.source, self.source]).make_array([U.real, U.imag])
        V = op.apply_inverse_adjoint(rhs)
        return V.block(0) + V.block(1) * 1j

    def as_range_array(self, mu=None):
        real = self.real.as_range_array(mu=mu)
        imag = self.imag.as_range_array(mu=mu)
        return real + imag * 1j

    def as_source_array(self, mu=None):
        real = self.real.as_source_array(mu=mu)
        imag = self.imag.as_source_array(mu=mu)
        return real + imag * 1j

    def assemble(self, mu=None):
        return ComplexOperator(self.real.assemble(mu=mu),
                               self.imag.assemble(mu=mu))
