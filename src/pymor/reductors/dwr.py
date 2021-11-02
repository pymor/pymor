# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from numbers import Number

from pymor.core.base import ImmutableObject, BasicObject
from pymor.algorithms.projection import project
from pymor.operators.interface import Operator
from pymor.operators.constructions import VectorOperator
from pymor.reductors.coercive import CoerciveRBReductor, CoerciveRBEstimator
from pymor.reductors.residual import ResidualOperator


class DWRCoerciveRBReductor(BasicObject):
    """Reduced Basis reductor for |StationaryModels| with coercive linear operator.

    This class can be used as a replacement for
    :class:`~pymor.reductors.coercive.CoerciveRBReductor` for a corrected output
    functional with the DWR approach. (see :cite:`Haa17` (Definition 2.26)).
    This also enables a DWR based error estimator for the corrected output functional.
    The DWR approach requires a dual problems for every output dimension of the output functional
    Each dual problem is then constructed with the dual operator and the appropriate component of
    the output functional as right hand side. See also :meth:`~pymor.reductors.dwr.dual_model`.

    Parameters
    ----------
    fom
        The |Model| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
    product
        Inner product for the orthonormalization of `RB`, the projection of the
        |Operators| given by `vector_ranged_operators` and for the computation of
        Riesz representatives of the residual. If `None`, the Euclidean product is used.
    coercivity_estimator
        `None` or a |Parameterfunctional| returning a lower bound for the coercivity
        constant of the given problem. Note that the computed error estimate is only
        guaranteed to be an upper bound for the error when an appropriate coercivity
        estimate is specified.
    operator_is_symmetric
        If the operator of `fom` is symmetric (in theory), it can make sense to consider
        the same operator also for the adjoint case for the dual models. In this case
        `operator_is_symmetric` as `True`, means to use the same operator for both the
        primal as well as for the dual model. If `False` the adjoint operator is used.
    dual_bases
        List of |VectorArrays| contraining reduced basis for the dual models that are
        constructed with :meth:`~pymor.reductors.dwr.dual_model`, where each entry
        of the list corresponds to the dimensions of the output functional.
    """

    def __init__(self, fom, RB=None, product=None, coercivity_estimator=None,
                 operator_is_symmetric=False, dual_bases=None, check_orthonormality=None,
                 check_tol=None, assemble_error_estimate=True, assemble_output_error_estimate=True):
        self.__auto_init(locals())
        self._last_rom = None
        self.bases = dict(RB=RB)

        if dual_bases is not None:
            assert len(dual_bases) == fom.dim_output

        self.primal_reductor = CoerciveRBReductor(fom, RB, product, coercivity_estimator, check_tol,
                                                  assemble_error_estimate,
                                                  assemble_output_error_estimate=False)
        self.dual_reductors = []
        assert (fom.output_functional is not None and fom.output_functional.linear), \
            'The features of the DWR reductor cannot be used, you should use CoerciveRBReductor instead.'

        # either needed for estimation or just for the corrected output
        for d in range(fom.dim_output):
            # construct dual model
            dual_model = self.dual_model(fom, d, operator_is_symmetric)
            # choose dual basis
            dual_basis = dual_bases[d]
            # define dual reductors
            dual_reductor = CoerciveRBReductor(dual_model, dual_basis, product, coercivity_estimator,
                                               check_orthonormality, check_tol)
            self.dual_reductors.append(dual_reductor)

    def reduce(self, dims=None):
        if dims is None:
            dims = {k: len(v) for k, v in self.bases.items()}
        if isinstance(dims, Number):
            dims = {k: dims for k in self.bases}
        if set(dims.keys()) != set(self.bases.keys()):
            raise ValueError(f'Must specify dimensions for {set(self.bases.keys())}')
        for k, d in dims.items():
            if d < 0:
                raise ValueError(f'Reduced state dimension must be larger than zero {k}')
            if d > len(self.bases[k]):
                raise ValueError(f'Specified reduced state dimension larger than reduced basis {k}')

        if self._last_rom is None or any(dims[b] > self._last_rom_dims[b] for b in dims):
            self._last_rom = self._reduce()
            self._last_rom_dims = {k: len(v) for k, v in self.bases.items()}

        if dims == self._last_rom_dims:
            return self._last_rom
        else:
            return self._reduce_to_subbasis(dims)

    def _reduce(self):
        primal_rom = self.primal_reductor.reduce()

        # reduce dual models
        reduced_dual_models = [red.reduce() for red in self.dual_reductors]

        # build corrected output
        dual_projected_primal_residuals = []
        for dual_reductor in self.dual_reductors:
            dual_basis = dual_reductor.bases['RB']
            op = project(self.fom.operator, dual_basis, self.bases['RB'])
            rhs = project(self.fom.rhs, dual_basis, None)
            primal_residual = ResidualOperator(op, rhs, name='dual_projected_residual')
            dual_projected_primal_residuals.append(primal_residual)
        corrected_output = CorrectedOutputFunctional(primal_rom.output_functional,
                                                     reduced_dual_models,
                                                     dual_projected_primal_residuals)

        # build error estimator
        dual_estimators = [dual_reductor.assemble_error_estimator()
                           for dual_reductor in self.dual_reductors]
        error_estimator = DWRCoerciveRBEstimator(primal_rom.error_estimator, dual_estimators,
                                                 reduced_dual_models, self.dual_reductors)

        # build rom
        rom = primal_rom.with_(output_functional=corrected_output, error_estimator=error_estimator)
        return rom

    def _reduce_to_subbasis(self, dims):
        projected_operators = self.project_operators_to_subbasis(dims)
        error_estimator = self.assemble_error_estimator_for_subbasis(dims)
        rom = self.build_rom(projected_operators, error_estimator)
        rom = rom.with_(name=f'{self.fom.name}_reduced')
        rom.disable_logging()
        return rom

    def build_rom(self, projected_operators, error_estimator):
        rom = super().build_rom(projected_operators, error_estimator)
        # replace the output functional by the corrected output functional
        corrected_output = self.build_corrected_output(rom, rom.solution_space.dim)
        rom = rom.with_(output_functional=corrected_output)
        return rom

    def build_corrected_output(self, rom, dim):
        dual_projected_primal_residuals = []
        for dual_reductor in self.dual_reductors:
            dual_basis = dual_reductor.bases['RB'][:dim]
            op = project(self.fom.operator, dual_basis, self.bases['RB'][:dim])
            rhs = project(self.fom.rhs, dual_basis, None)
            primal_residual = ResidualOperator(op, rhs, name='dual_projected_residual')
            dual_projected_primal_residuals.append(primal_residual)
        if dim < self.reduced_dual_models[0].solution_space.dim:
            # dimension does not fit anymore
            self.reduced_dual_models = [red.reduce(dim) for red in self.dual_reductors]
        return CorrectedOutputFunctional(rom.output_functional, self.reduced_dual_models,
                                         dual_projected_primal_residuals)

    def assemble_error_estimator(self):
        residual = self.residual_reductor.reduce()
        dual_estimators = [dual_reductor.assemble_error_estimator()
                           for dual_reductor in self.dual_reductors]
        primal_estimator = CoerciveRBEstimator(residual, tuple(self.residual_reductor.residual_range_dims),
                                               self.coercivity_estimator)
        self.reduced_dual_models = [red.reduce() for red in self.dual_reductors]
        error_estimator = DWRCoerciveRBEstimator(primal_estimator, dual_estimators, self.reduced_dual_models,
                                                 self.dual_reductors)
        return error_estimator

    @classmethod
    def dual_model(cls, model, dim=0, operator_is_symmetric=False):
        """Return dual model with the output as right hand side.

        See :cite:`Haa17` (Definition 2.26)

        Parameters
        ----------
        model
            The |Model| for which to construct the dual model
        dim
            The dimension of the `fom.output_functional` for which the dual model is to be built.
        operator_is_symmetric
            If `True`, `fom.operator` is used for the dual problem.
            This is only feasable if the operator is symmetric (in theory).
            If `False` the adjoint `fom.operator.H` is used instead.

        Returns
        -------
        A |Model| with the adjoint operator and the corresponding right hand side
        """
        assert 0 <= dim < model.dim_output
        e_i_vec = model.output_functional.range.from_numpy(np.eye(1, model.dim_output, dim))
        dual_rhs = - model.output_functional.H @ VectorOperator(e_i_vec)
        dual_operator = model.operator if operator_is_symmetric else model.operator.H
        dual_model = model.with_(operator=dual_operator, rhs=dual_rhs,
                                 output_functional=None, name=model.name + '_dual')
        return dual_model

    def project_operators_to_subbasis(self, dims):
        self._last_rom = self._last_rom.with_(
            output_functional=self._last_rom.output_functional.output_functional)
        return super().project_operators_to_subbasis(dims)

    def assemble_error_estimator_for_subbasis(self, dims):
        return self._last_rom.error_estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)


class DWRCoerciveRBEstimator(ImmutableObject):
    """Instantiated by :class:`DWRCoerciveRBReductor`.

    Not to be used directly.
    """

    def __init__(self, primal_estimator, dual_estimators, dual_models, dual_reductors):
        self.__auto_init(locals())

    def estimate_error(self, U, mu, m):
        return self.primal_estimator.estimate_error(U, mu, m)

    def estimate_output_error(self, U, mu, m):
        est_pr = self.estimate_error(U, mu, m)
        est_dus = []
        for d in range(m.dim_output):
            dual_solution = self.dual_models[d].solve(mu)
            est_dus.append(self.dual_estimators[d].estimate_error(dual_solution, mu, m))
        return (est_pr * est_dus).T

    def restricted_to_subbasis(self, dim, m):
        primal_estimator = self.primal_estimator.restricted_to_subbasis(dim, m)
        dual_models = [red.reduce(dim) for red in self.dual_reductors]
        dual_estimators = [dual_estimator.restricted_to_subbasis(dim, m) for dual_estimator
                           in self.dual_estimators]
        return DWRCoerciveRBEstimator(primal_estimator, dual_estimators, dual_models, self.dual_reductors)


class CorrectedOutputFunctional(Operator):
    """|Operator| representing the corrected output functional from :cite:`Haa17` (Definition 2.26)

    Parameters
    ----------
    output_functional
        Original output_functional
    dual_models
        dual models for the corrected output
    dual_projected_primal_residuals
        The evaluated primal residuals
    """

    linear = False

    def __init__(self, output_functional, dual_models, dual_projected_primal_residuals):
        self.__auto_init(locals())
        self.source = output_functional.source
        self.range = output_functional.range

    def apply(self, solution, mu=None):
        # compute corrected output functional
        output = self.output_functional.apply(solution, mu=mu).to_numpy()
        correction = np.empty((self.range.dim, len(solution)))
        for d, (dual_m, dual_res) in enumerate(zip(self.dual_models, self.dual_projected_primal_residuals)):
            dual_solution = dual_m.solve(mu)
            dual_correction = dual_res.apply2(dual_solution, solution, mu)
            correction[d] = dual_correction
        corrected_output = output + correction.T
        return self.range.from_numpy(corrected_output)
