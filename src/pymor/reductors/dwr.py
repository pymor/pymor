# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from numbers import Number

from pymor.core.base import ImmutableObject, BasicObject
from pymor.algorithms.projection import project
from pymor.operators.interface import Operator
from pymor.operators.constructions import VectorOperator
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.reductors.residual import ResidualOperator


class DWRCoerciveRBReductor(BasicObject):
    """Reduced Basis reductor for |StationaryModels| with coercive linear operator

    This class can be used as a replacement for
    :class:`~pymor.reductors.coercive.CoerciveRBReductor` for a corrected output
    functional with the DWR approach. (see :cite:`Haa17` (Definition 2.31)).
    This also enables a DWR based error estimator for the corrected output functional.
    The DWR approach requires a dual problem for every dimension of the output functional.
    Each dual problem is defined by the dual operator and the corresponding component of
    the output functional as right hand side. See also :meth:`~pymor.reductors.dwr.dual_model`.

    Parameters
    ----------
    fom
        The |Model| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
    product
        See :class:`~pymor.reductors.coercive.CoerciveRBReductor`.
    coercivity_estimator
        See :class:`~pymor.reductors.coercive.CoerciveRBReductor`.
    operator_is_symmetric
        If the operator of `fom` is symmetric (in theory and before boundary treatment),
        it can make sense to consider the same operator also for the adjoint case
        for the dual models. In this case `operator_is_symmetric` as `True`,
        means to use the same operator for both the primal as well as for the dual model.
        If `False` the adjoint operator is used.
    dual_bases
        List of |VectorArrays| containing the reduced basis for the dual models that are
        constructed with :meth:`~pymor.reductors.dwr.dual_model`, where each entry
        of the list corresponds to the dimensions of the output functional.
        If `dual_bases` is `None`, the primal bases are used.
    check_orthonormality
        See :class:`~pymor.reductors.basic.ProjectionBasedReductor`.
    check_tol
        See :class:`~pymor.reductors.basic.ProjectionBasedReductor`.
    """

    def __init__(self, fom, primal_basis=None, product=None, coercivity_estimator=None,
                 operator_is_symmetric=False, dual_bases=None, check_orthonormality=None,
                 check_tol=None):
        self.__auto_init(locals())
        self._last_rom = None

        if dual_bases is not None:
            assert len(dual_bases) == fom.dim_output

        self.primal_reductor = CoerciveRBReductor(fom, primal_basis, product,
                                                  coercivity_estimator, check_tol)
        self.dual_reductors = []
        assert (fom.output_functional is not None and fom.output_functional.linear), \
            'The features of the DWR reductor cannot be used, ' + \
            'please use CoerciveRBReductor instead.'

        # either needed for estimation or just for the corrected output
        for d in range(fom.dim_output):
            # construct dual model
            dual_model = self.dual_model(fom, d, operator_is_symmetric)
            # choose dual basis
            if dual_bases is not None:
                dual_basis = dual_bases[d]
            else:
                dual_basis = primal_basis
            # define dual reductors (with None as coercivity_estimator)
            dual_reductor = CoerciveRBReductor(dual_model, dual_basis, product, None,
                                               check_orthonormality, check_tol)
            self.dual_reductors.append(dual_reductor)

    def reduce(self, dim=None):
        dim = dim or len(self.primal_basis)
        assert isinstance(dim, Number)
        if dim < 0:
            raise ValueError('Reduced state dimension must be larger than zero')
        if dim > len(self.primal_basis) and all([dim > len(dual_basis) for dual_basis in self.dual_bases]):
            raise ValueError('Specified reduced state dimension larger than reduced basis')

        if self._last_rom is None or dim > self._last_rom_dim:
            self._last_rom = self._reduce()
            self._last_rom_dim = dim

        if dim == self._last_rom_dim:
            return self._last_rom
        else:
            return self._reduce_to_subbasis(dim)

    def _reduce(self):
        primal_rom = self.primal_reductor.reduce()

        # reduce dual models with most possible basis functions
        dual_roms = [red.reduce() for red in self.dual_reductors]

        # build corrected output
        corrected_output = self.build_corrected_output(primal_rom, dual_roms)

        # build error estimator
        dual_estimators = [dual_reductor.assemble_error_estimator()
                           for dual_reductor in self.dual_reductors]
        error_estimator = DWRCoerciveRBEstimator(primal_rom.error_estimator, dual_estimators,
                                                 dual_roms)

        # build rom
        rom = primal_rom.with_(output_functional=corrected_output, error_estimator=error_estimator)
        return rom

    def _reduce_to_subbasis(self, dim):
        primal_rom = self.primal_reductor.reduce(dim)
        dual_roms = [red.reduce(dim) for red in self.dual_reductors]
        corrected_output = self.build_corrected_output(primal_rom, dual_roms, dim)
        error_estimator = self.assemble_error_estimator_for_subbasis(dual_roms, dim)
        rom = primal_rom.with_(output_functional=corrected_output, error_estimator=error_estimator)
        return rom

    def build_corrected_output(self, primal_rom, dual_roms, dim=None):
        # Note: if dim==None, then the dual and primal basis size can differ!
        # This is the case if reduce() is called without dim specified
        dual_projected_primal_residuals = []
        for dual_reductor in self.dual_reductors:
            dual_basis = dual_reductor.bases['RB'][:dim]
            op = project(self.fom.operator, dual_basis, self.primal_basis[:dim])
            rhs = project(self.fom.rhs, dual_basis, None)
            primal_residual = ResidualOperator(op, rhs, name='dual_projected_residual')
            dual_projected_primal_residuals.append(primal_residual)
        return CorrectedOutputFunctional(primal_rom.output_functional, dual_roms,
                                         dual_projected_primal_residuals)

    @classmethod
    def dual_model(cls, model, dim=0, operator_is_symmetric=False):
        """Return dual model with the output as right hand side.

        The dual equation is defined as to find the solution p such that

            a(q, p) = - l_dim(q),       for all q,

        where l_dim denotes the dim-th component of the output functional l.
        See :cite:`Haa17` (Definition 2.26)

        Parameters
        ----------
        model
            The |Model| for which to construct the dual model
        dim
            The dimension of the `fom.output_functional` for which the dual model
            is to be built.
        operator_is_symmetric
            If `True`, `fom.operator` is used for the dual operator of a(., .).
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

    def assemble_error_estimator_for_subbasis(self, dual_roms, dim):
        return self._last_rom.error_estimator.restricted_to_subbasis(dual_roms, dim, m=self._last_rom)

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self.primal_reductor.reconstruct(u)

    def extend_basis(self, U, Ps, method='gram_schmidt', copy_U=True):
        self.primal_reductor.extend_basis(U, method=method, copy_U=copy_U)
        for i, P in enumerate(Ps):
            self.dual_reductors[i].extend_basis(P, method=method, copy_U=copy_U)


class DWRCoerciveRBEstimator(ImmutableObject):
    """Instantiated by :class:`DWRCoerciveRBReductor`.

    Not to be used directly.
    """

    def __init__(self, primal_estimator, dual_estimators, dual_models):
        self.__auto_init(locals())

    def estimate_error(self, U, mu, m):
        return self.primal_estimator.estimate_error(U, mu, m)

    def estimate_output_error(self, U, mu, m, return_vector=False):
        est_pr = self.estimate_error(U, mu, m)
        est_dus = []
        for d in range(m.dim_output):
            dual_solution = self.dual_models[d].solve(mu)
            est_dus.append(self.dual_estimators[d].estimate_error(dual_solution, mu, m))
        # since dual models have a constant coercivity estimator,
        # we do not have to multiply it here.
        ret = (est_pr * est_dus).T
        if return_vector:
            return ret
        else:
            return np.linalg.norm(ret)

    def restricted_to_subbasis(self, dual_roms, dim, m):
        primal_estimator = self.primal_estimator.restricted_to_subbasis(dim, m)
        dual_estimators = [dual_estimator.restricted_to_subbasis(dim, m) for dual_estimator
                           in self.dual_estimators]
        return DWRCoerciveRBEstimator(primal_estimator, dual_estimators, dual_roms)


class CorrectedOutputFunctional(Operator):
    """|Operator| representing the corrected output functional from :cite:`Haa17` (Definition 2.31)

    Parameters
    ----------
    output_functional
        Original output_functional
    dual_models
        Dual models for the corrected output, see :meth:`~pymor.reductors.dwr.dual_model`
    dual_projected_primal_residuals
        The primal residuals projected on the dual space (in the first argument) and on the
        primal space (in the second argument)
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
