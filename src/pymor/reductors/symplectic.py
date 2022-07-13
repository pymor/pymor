# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np
from pymor.algorithms.projection import project
from pymor.algorithms.simplify import contract, expand
from pymor.algorithms.symplectic import SymplecticBasis
from pymor.core.base import BasicObject
from pymor.models.symplectic import (BaseQuadraticHamiltonianModel,
                                     QuadraticHamiltonianModel)
from pymor.operators.constructions import IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator


class QuadraticHamiltonianRBReductor(BasicObject):
    """Symplectic Galerkin projection of a |QuadraticHamiltonianModel|.

    Parameters
    ----------
    fom
        The full order |QuadraticHamiltonianModel| to reduce.
    RB
        A |SymplecticBasis| prescribing the basis vectors.
    """

    def __init__(self, fom, RB=None):
        assert isinstance(fom, QuadraticHamiltonianModel)
        if not isinstance(fom.mass, IdentityOperator):
            raise NotImplementedError
        if fom.products:
            raise NotImplementedError

        RB = SymplecticBasis(phase_space=fom.operator.source) if RB is None else RB
        assert isinstance(RB, SymplecticBasis)
        assert RB.phase_space == fom.solution_space

        self.fom = fom
        self.RB = RB
        self.bases = {'RB': RB}

    def extend_basis(self, U, method='svd_like', modes=2, copy_U=False):
        self.RB.extend(U, method=method, modes=modes)

    def reduce(self, dims=None):
        with self.logger.block('Operator projection ...'):
            if isinstance(dims, dict):
                dim = dims.get('RB', None)
            elif isinstance(dims, Number) or dims is None:
                dim = dims
            else:
                raise NotImplementedError()

            if dim is None:
                dim = len(self.RB) * 2
            assert dim % 2 == 0, 'dim has to be even'

            fom = self.fom
            RB = self.RB[:dim//2]
            tsiRB = RB.transposed_symplectic_inverse().to_array()
            RB = RB.to_array()

            projected_operators = {
                'H_op':              project(fom.H_op, RB, RB),
                'h':                 project(fom.h, RB, None),
                'initial_data':      project(fom.initial_data, tsiRB, None),
                'output_functional': project(fom.output_functional, None, RB) if fom.output_functional else None
            }

        with self.logger.block('Building ROM ...'):
            rom = ReducedQuadraticHamiltonianModel(
                fom.T,
                time_stepper=fom.time_stepper,
                num_values=fom.num_values,
                name=None if fom.name is None else 'reduced_' + fom.name,
                **projected_operators
            )
        return rom

    def reconstruct(self, u):
        return self.RB[:u.dim//2].lincomb(u.to_numpy())


class ReducedQuadraticHamiltonianModel(BaseQuadraticHamiltonianModel):
    """A general reduced quadratic Hamiltonian system.

    In contrast to the |QuadraticHamiltonianModel|, the reduced model does not rely on a
    |BlockVectorSpace| as `phase_space`.
    """

    def __init__(self, T, initial_data, H_op, h=None, time_stepper=None, nt=None, num_values=None,
                 output_functional=None, visualizer=None, name=None):

        # generate J as NumpyMatrixOperator
        assert H_op.source.dim % 2 == 0
        n = H_op.source.dim // 2
        J = NumpyMatrixOperator(
            np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]]))

        super().__init__(T, initial_data, J, H_op, h, time_stepper, nt, num_values,
                         output_functional, visualizer, name)

        self.operator = contract(expand(self.operator))
