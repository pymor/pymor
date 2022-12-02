# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from numbers import Number

from pymor.algorithms.projection import project
from pymor.core.base import ImmutableObject, BasicObject
from pymor.operators.block import BlockColumnOperator
from pymor.operators.constructions import ConcatenationOperator, NumpyConversionOperator, VectorOperator
from pymor.operators.interface import Operator
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.reductors.residual import ResidualOperator


class DWRCoerciveRBReductor(BasicObject):
    """Reduced Basis reductor for |StationaryModels| with coercive linear operator

    This class can be used as a replacement for
    :class:`~pymor.reductors.coercive.CoerciveRBReductor` to obtain a corrected reduced
    output functional with the DWR approach (see :cite:`Haa17` (Definition 2.31, Proposition 2.32)).
    This also implements a DWR-based error estimator for the corrected output functional.
    The DWR approach requires the reduction of a dual problem for every dimension of the output
    functional. Each dual problem is defined by the dual operator and the corresponding component
    of the output functional as right-hand side. See also :meth:`~pymor.reductors.dwr.dual_model`.

    Parameters
    ----------
    fom
        The |Model| which is to be reduced.
    dual_foms
        List of the dual |Models| that correspond to each dimension of the output_functional.
        If `dual_foms` is `None`, the default dual models are constructed
        by :meth:`~pymor.reductors.dwr.create_dual_model`, assuming a fully discrete perspective.
    primal_RB
        |VectorArray| containing the reduced basis on which to project the fom.
    dual_RBs
        List of |VectorArrays| containing the reduced bases on which to project the `dual_foms`,
        where each entry of the list corresponds to the dimensions of the output functional.
        If `dual_bases` is `None`, the primal basis are used.
    product
        See :class:`~pymor.reductors.coercive.CoerciveRBReductor`.
    coercivity_estimator
        See :class:`~pymor.reductors.coercive.CoerciveRBReductor`.
    check_orthonormality
        See :class:`~pymor.reductors.basic.ProjectionBasedReductor`.
    check_tol
        See :class:`~pymor.reductors.basic.ProjectionBasedReductor`.
    """

    def __init__(self, fom, dual_foms=None, primal_RB=None, dual_RBs=None, product=None,
                 coercivity_estimator=None, check_orthonormality=None, check_tol=None):
        self.__auto_init(locals())
        self._last_rom = None

        if dual_RBs is not None:
            assert len(dual_RBs) == fom.dim_output
        assert (fom.output_functional is not None and fom.output_functional.linear), \
            'DWRCoerciveRBReductor requires a linear ouput functional. ' + \
            'Please use CoerciveRBReductor instead.'

        self.primal_reductor = CoerciveRBReductor(fom, RB=primal_RB, product=product,
                                                  coercivity_estimator=coercivity_estimator,
                                                  check_orthonormality=check_orthonormality,
                                                  check_tol=check_tol)
        # construct default dual models if not provided
        self.dual_foms = dual_foms or [self.create_dual_model(fom, d) for d in range(fom.dim_output)]
        self.dual_reductors = []
        # either needed for estimation or just for the corrected output
        for d in range(fom.dim_output):
            # construct dual model
            dual_model = self.dual_foms[d]
            # define dual reductors (with None as coercivity_estimator)
            dual_basis = primal_RB if dual_RBs is None else dual_RBs[d]
            dual_reductor = CoerciveRBReductor(dual_model, RB=dual_basis, product=product,
                                               coercivity_estimator=None,
                                               check_orthonormality=check_orthonormality,
                                               check_tol=check_tol)
            self.dual_reductors.append(dual_reductor)

    def reduce(self, primal_dim=None, dual_dims=None):
        if primal_dim is None:
            primal_dim = len(self.primal_reductor.bases['RB'])
        assert isinstance(primal_dim, Number)
        if isinstance(dual_dims, Number):
            dual_dims = [dual_dims for _ in range(self.fom.dim_output)]
        if dual_dims is None:
            dual_dims = [len(dual_reductor.bases['RB']) for dual_reductor in self.dual_reductors]
        assert isinstance(dual_dims, list)
        assert all(isinstance(dim, Number) for dim in dual_dims)
        if primal_dim < 0 or any(dim < 0 for dim in dual_dims):
            raise ValueError('Reduced state dimension must be non-negative')
        if primal_dim > len(self.primal_reductor.bases['RB']) or \
                any(dim > len(dual_reductor.bases['RB'])
                    for dim, dual_reductor in zip(dual_dims, self.dual_reductors)):
            raise ValueError('Specified reduced state dimension larger than reduced basis')

        dims = [primal_dim] + dual_dims
        if self._last_rom is None or any((dim > last_rom_dim)
                                         for dim, last_rom_dim in zip(dims, self._last_rom_dims)):
            self._last_rom = self._reduce()
            self._last_rom_dims = dims

        if dims == self._last_rom_dims:
            return self._last_rom
        else:
            return self._reduce_to_subbasis(primal_dim, dual_dims)

    def _reduce(self):
        with self.logger.block('Reducing primal FOM ...'):
            primal_rom = self.primal_reductor.reduce()

        with self.logger.block('Reducing dual FOM ...'):
            dual_roms = [red.reduce() for red in self.dual_reductors]

        with self.logger.block('Constructing DWR error estimator ...'):
            dual_estimators = [dual_rom.error_estimator for dual_rom in dual_roms]
            error_estimator = DWRCoerciveRBEstimator(primal_rom.error_estimator, dual_estimators, dual_roms)

        with self.logger.block('Building corrected output ...'):
            corrected_output = self._build_corrected_output(primal_rom, dual_roms)

        with self.logger.block('Building ROM ...'):
            rom = primal_rom.with_(output_functional=corrected_output, error_estimator=error_estimator)

        return rom

    def _reduce_to_subbasis(self, primal_dim, dual_dims):
        primal_rom = self.primal_reductor.reduce(primal_dim)
        dual_roms = [red.reduce(dim) for red, dim in zip(self.dual_reductors, dual_dims)]
        corrected_output = self._build_corrected_output(primal_rom, dual_roms, primal_dim, dual_dims)
        error_estimator = self.assemble_error_estimator_for_subbasis(dual_roms, primal_dim, dual_dims)
        rom = primal_rom.with_(output_functional=corrected_output, error_estimator=error_estimator)
        return rom

    def _build_corrected_output(self, primal_rom, dual_roms, primal_dim=None, dual_dims=None):
        dual_projected_primal_residuals = []
        dual_dims = dual_dims or [None for _ in range(self.fom.dim_output)]
        for dual_reductor, dual_dim in zip(self.dual_reductors, dual_dims):
            dual_basis = dual_reductor.bases['RB'][:dual_dim]
            op = project(self.fom.operator, dual_basis,
                         self.primal_reductor.bases['RB'][:primal_dim])
            rhs = project(self.fom.rhs, dual_basis, None)
            primal_residual = ResidualOperator(op, rhs, name='dual_projected_residual')
            dual_projected_primal_residuals.append(primal_residual)
        return CorrectedOutputFunctional(primal_rom.output_functional, dual_roms,
                                         dual_projected_primal_residuals)

    @classmethod
    def create_dual_model(cls, model, dim=0):
        r"""Return dual model with the output as right hand side.

        The dual equation is defined as to find the solution :math:`p` such that

        .. math::
            a(q, p) = - l_d(q),\qquad\text{for all }q,

        where :math:`l_d` denotes the :math:`d`-th component of the output functional :math:`l`.
        See :cite:`Haa17` (Definition 2.31).

        Parameters
        ----------
        model
            The |Model| for which to construct the dual model
        dim
            The dimension of the `fom.output_functional` for which the dual model
            is to be built.

        Returns
        -------
        A |Model| with the adjoint operator and the corresponding right-hand side
        """
        assert 0 <= dim < model.dim_output
        output = model.output_functional
        if (isinstance(output, ConcatenationOperator) and len(output.operators) == 2
                and isinstance(output.operators[0], NumpyConversionOperator)
                and isinstance(output.operators[1], BlockColumnOperator)):
            # the output_functional has the same structure as built by the discretizers
            # which allows for unblocking the right-hand side efficiently.
            dual_rhs = - output.operators[1].blocks[dim][0].H
        else:
            # case where model.dim_output == 1 and
            # more general case without using the structure of BlockColumnOperator
            if model.dim_output > 1:
                model.logger.warn('Using inefficient concatenation for the right-hand side')
            e_i_vec = model.output_functional.range.from_numpy(np.eye(1, model.dim_output, dim))
            dual_rhs = - output.H @ VectorOperator(e_i_vec) if model.dim_output > 1 else - output.H
        dual_operator = model.operator.H
        dual_model = model.with_(operator=dual_operator, rhs=dual_rhs,
                                 output_functional=None, name=model.name + '_dual')
        return dual_model

    def assemble_error_estimator_for_subbasis(self, dual_roms, primal_dim, dual_dims):
        return self._last_rom.error_estimator.restricted_to_subbasis(dual_roms, primal_dim,
                                                                     dual_dims, m=self._last_rom)

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self.primal_reductor.reconstruct(u)

    def extend_basis(self, U, Ps, method='gram_schmidt', copy=True):
        self.primal_reductor.extend_basis(U, method=method, copy_U=copy)
        for i, P in enumerate(Ps):
            self.dual_reductors[i].extend_basis(P, method=method, copy_U=copy)


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
        ret = (est_pr * est_dus).T
        return ret if return_vector else np.linalg.norm(ret)

    def restricted_to_subbasis(self, dual_roms, primal_dim, dual_dims, m):
        primal_estimator = self.primal_estimator.restricted_to_subbasis(primal_dim, m)
        dual_estimators = [dual_estimator.restricted_to_subbasis(dim, m) for
                           dual_estimator, dim in zip(self.dual_estimators, dual_dims)]
        return DWRCoerciveRBEstimator(primal_estimator, dual_estimators, dual_roms)


class CorrectedOutputFunctional(Operator):
    """|Operator| representing the corrected output functional from :cite:`Haa17` (Definition 2.31)

    Parameters
    ----------
    output_functional
        Original output_functional
    dual_models
        Dual models for the corrected output, see :meth:`~pymor.reductors.dwr.create_dual_model`
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
