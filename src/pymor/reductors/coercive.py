# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.base import ImmutableObject
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import LincombOperator, induced_norm, VectorOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.interface import Operator
from pymor.reductors.basic import StationaryRBReductor
from pymor.reductors.residual import ResidualReductor, ResidualOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class CoerciveRBReductor(StationaryRBReductor):
    """Reduced Basis reductor for |StationaryModels| with coercive linear operator.

    The only addition to :class:`~pymor.reductors.basic.StationaryRBReductor` is an error
    estimator which evaluates the dual norm of the residual with respect to a given inner
    product and an output error estimator. For the reduction of the residual we use
    :class:`~pymor.reductors.residual.ResidualReductor` for improved numerical stability
    :cite:`BEOR14`.

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
        If `assemble_output_error_estimate` is `True`, the DWR estimator can either be
        build with the operator itself (if the operator is symmetric, or with the adjoint
        operator. For the symmetric case `operator_is_symmetric` is to be set as `True`.
    dual_basis
        If `operator_is_symmetric` is `False` or if the output functional of the |Model|
        differs substantially from the right hand side of the |Model|, it makes sense to
        provide a reduced basis for the dual problems
        (see :classmethod:`~pymor.reductors.coercive.CoerciveRBReductor.dual_model` for details)
    """

    def __init__(self, fom, RB=None, product=None, coercivity_estimator=None,
                 operator_is_symmetric=False, dual_bases=None, check_orthonormality=None,
                 check_tol=None, assemble_error_estimate=True, assemble_output_error_estimate=True):
        super().__init__(fom, RB, product=product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol, assemble_error_estimate=assemble_error_estimate,
                         assemble_output_error_estimate=assemble_output_error_estimate)
        self.coercivity_estimator = coercivity_estimator
        self.residual_reductor = ResidualReductor(self.bases['RB'], self.fom.operator, self.fom.rhs,
                                                  product=product, riesz_representatives=True)
        self.corrected_output = False
        self.dual_bases = dual_bases
        if fom.output_functional is not None:
            if fom.output_functional.linear and (assemble_output_error_estimate or dual_bases is not None):
                # either needed for estimation or just for the corrected output
                if dual_bases is not None:
                    # corrected output only makes sense if the basis differs from the dual basis
                    assert len(dual_bases) == fom.dim_output
                    self.corrected_output = True
                self.dual_residual_reductors, self.dual_projected_primal_residuals = [], []
                self.projected_dual_operators, self.projected_dual_rhss = [], []
                output = self.fom.output_functional
                for d in range(fom.dim_output):
                    # choose dual basis
                    if dual_bases is not None:
                        dual_basis = dual_bases[d]
                    else:
                        dual_basis = self.bases['RB']
                    # choose operator
                    if operator_is_symmetric:
                        dual_operator = self.fom.operator
                    else:
                        if dual_bases is None:
                            self.logger.warn('You are using a wrong basis for the adjoint operator. '
                                             'If you are sure that your operator is symmetric (in theory), '
                                             'you can set `operator_is_symmetric = True`. If your operator '
                                             'is not symmetric, you should provide a dual basis via `dual_bases`.')
                        dual_operator = self.fom.operator.H
                    # construct dual rhs
                    e_i = np.zeros(fom.dim_output)
                    e_i[d] = 1
                    e_i_vec = output.range.from_numpy(e_i)
                    restricted_output = - output.H @ VectorOperator(e_i_vec)
                    # define dual residual
                    self.dual_residual_reductors.append(ResidualReductor(dual_basis, dual_operator,
                                                                         restricted_output, product=product,
                                                                         riesz_representatives=True))

    def project_operators(self):
        projected_operators = super().project_operators()

        if hasattr(self, 'dual_residual_reductors'):
            # either the corrected output needs to be built or the output error estimate or both
            self.dual_projected_primal_residuals, self.reduced_dual_models = [], []
            for dual_residual in self.dual_residual_reductors:
                basis = dual_residual.RB
                projected_dual_operator = project(dual_residual.operator, basis, basis)
                projected_dual_rhs = project(dual_residual.rhs, basis, None)
                dual_model = StationaryModel(projected_dual_operator, projected_dual_rhs,
                                             name=self.fom.name + '_reduced_dual')
                self.reduced_dual_models.append(dual_model)
                if self.dual_bases is not None:
                    op = project(self.fom.operator, basis, self.bases['RB'])
                    rhs = project(self.fom.rhs, basis, None)
                    residual = ResidualOperator(op, rhs, name='dual_projected_residual')
                    self.dual_projected_primal_residuals.append(residual)

            if self.dual_bases is not None:
                # We replace the output functional by the corrected output functional
                # this will only be used if `dual_bases` are given in the reductor.
                standard_output = projected_operators['output_functional']
                projected_operators['output_functional'] = CorrectedOutputFunctional(
                    standard_output, self.reduced_dual_models, self.dual_projected_primal_residuals)

        return projected_operators

    def project_operators_to_subbasis(self, dims):
        if self.corrected_output:
            # this is a hack because there is no project rule for CorrectedOutputFunctional defined
            # TODO: define a project rule without circular imports.
            corrected_output = self._last_rom.output_functional
            self._last_rom = self._last_rom.with_(output_functional=None)
        projected_operators = super().project_operators_to_subbasis(dims)
        if self.corrected_output:
            dim = dims['RB']
            projected_operators['output_functional'] = corrected_output.reduce_to_subbasis(dim, dim)
            self._last_rom = self._last_rom.with_(output_functional=corrected_output)
        return projected_operators

    def assemble_error_estimator(self):
        residual = self.residual_reductor.reduce()
        dual_ress, dual_range_dims, reduced_dual_models = self.prepare_dwr_output_error_estimator()
        error_estimator = CoerciveRBEstimator(residual, tuple(self.residual_reductor.residual_range_dims),
                                              self.coercivity_estimator, dual_ress, dual_range_dims,
                                              reduced_dual_models)
        return error_estimator

    @classmethod
    def dual_model(cls, fom, dim=0, operator_is_symmetric=False):
        """Return dual model with the output as right hand side.

        See :cite:`Haa17` (Definition 2.26)

        Parameters
        ----------
        fom
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

        assert 0 <= dim < fom.dim_output
        e_i = np.zeros(fom.dim_output)
        e_i[dim] = 1
        e_i_vec = fom.output_functional.range.from_numpy(e_i)
        dual_rhs = - fom.output_functional.H @ VectorOperator(e_i_vec)
        dual_operator = fom.operator if operator_is_symmetric else fom.operator.H
        dual_fom = fom.with_(operator=dual_operator, rhs=dual_rhs,
                             output_functional=None, name=fom.name + '_dual')
        return dual_fom

    def prepare_dwr_output_error_estimator(self):
        """Prepare the output error estimator with the DWR approach.

        See :cite:`Haa17` (Proposition 2.27).
        If the no (corrected) output needs to be built or no estimation is required,
        this code returns empty defaults.
        """

        dual_residuals, dual_range_dims, reduced_dual_models = [], [], None
        if hasattr(self, 'dual_residual_reductors'):
            if self.assemble_output_error_estimate:
                for dual_residual_reductor in self.dual_residual_reductors:
                    dual_residuals.append(dual_residual_reductor.reduce())
                    dual_range_dims.append(tuple(dual_residual_reductor.residual_range_dims))
            reduced_dual_models = self.reduced_dual_models
        # go back to None if nothing happened (this can also happen when the for loop started)
        dual_range_dims = None if len(dual_range_dims) == 0 else dual_range_dims
        return dual_residuals, dual_range_dims, reduced_dual_models

    def assemble_error_estimator_for_subbasis(self, dims):
        return self._last_rom.error_estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)


class CorrectedOutputFunctional(Operator):
    """|Operator| representing the corrected output functional from :cite:`Haa17` (Definition 2.26)

    Parameters
    ----------
    output_functional
        Original output_functional
    dual_models
        All dual models that are required for the corrected output
    dual_projected_primal_residuals
        The evaluated primal residuals
    """

    def __init__(self, output_functional, dual_models, dual_projected_primal_residuals):
        self.__auto_init(locals())
        self.linear = False
        self.source = output_functional.source
        self.range = output_functional.range

    def apply(self, solution, mu=None):
        # compute corrected output functional
        output = self.output_functional.apply(solution, mu=mu).to_numpy()
        dual_corrections = []
        for dual_m, dual_res in zip(self.dual_models, self.dual_projected_primal_residuals):
            dual_solution = dual_m.solve(mu)
            dual_correction = dual_res.apply2(dual_solution, solution, mu)
            dual_corrections.append(dual_correction)
        return self.range.from_numpy((output + dual_corrections)[0])

    def reduce_to_subbasis(self, dim_range, dim_source):
        dual_projected_primal_residuals, reduced_dual_models = [], []
        for dual_model, dual_residuals in zip(self.dual_models, self.dual_projected_primal_residuals):
            projected_dual_operator = project_to_subbasis(dual_model.operator, dim_range, dim_source)
            projected_dual_rhs = project_to_subbasis(dual_model.rhs, dim_range, None)
            restricted_dual_model = StationaryModel(projected_dual_operator, projected_dual_rhs,
                                                    name=dual_model.name + '_restricted')
            reduced_dual_models.append(restricted_dual_model)

            op = project_to_subbasis(dual_residuals.operator, dim_range, dim_source)
            rhs = project_to_subbasis(dual_residuals.rhs, dim_range, None)
            residual = ResidualOperator(op, rhs, name='dual_projected_residual_restricted')
            dual_projected_primal_residuals.append(residual)

        standard_output = project_to_subbasis(self.output_functional, None, dim_source)
        return CorrectedOutputFunctional(standard_output, reduced_dual_models,
                                         dual_projected_primal_residuals)


class CoerciveRBEstimator(ImmutableObject):
    """Instantiated by :class:`CoerciveRBReductor`.

    Not to be used directly.
    """

    def __init__(self, residual, residual_range_dims, coercivity_estimator, dual_residuals=None,
                 dual_residuals_range_dims=None, dual_models=None):
        self.__auto_init(locals())

    def estimate_error(self, U, mu, m):
        est = self.residual.apply(U, mu=mu).norm()
        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)
        return est

    def estimate_output_error(self, U, mu, m):
        assert m.output_functional is not None and self.dual_models is not None
        assert len(self.dual_residuals) == m.dim_output
        est_pr = self.estimate_error(U, mu, m)
        est_dus = []
        for d in range(m.dim_output):
            dual_solution = self.dual_models[d].solve(mu)
            est_dus.append(self.dual_residuals[d].apply(dual_solution, mu=mu).norm())
        return (est_pr * est_dus).T

    def restricted_to_subbasis(self, dim, m):
        if self.dual_models is not None:
            restricted_dual_models = []
            for dual_m in self.dual_models:
                restricted_dual_operator = project_to_subbasis(dual_m.operator, dim, dim)
                restricted_dual_rhs = project_to_subbasis(dual_m.rhs, dim, None)
                restricted_dual_m = StationaryModel(restricted_dual_operator,
                                                    restricted_dual_rhs,
                                                    name=dual_m.name + '_restricted')
                restricted_dual_models.append(restricted_dual_m)
        else:
            restricted_dual_models = None
        if self.residual_range_dims:
            residual_range_dims = self.residual_range_dims[:dim + 1]
            residual = self.residual.projected_to_subbasis(residual_range_dims[-1], dim)
            dual_residuals, dual_residuals_range_dims = [], []
            if self.dual_residuals_range_dims is not None:
                if len(self.dual_residuals_range_dims[0]) > 0:
                    assert len(self.dual_residuals_range_dims) == m.dim_output
                    assert len(self.dual_residuals) == m.dim_output
                    dual_residuals_range_dims = [res_range_dims[:dim + 1] for res_range_dims in
                                                 self.dual_residuals_range_dims]
                    dual_residuals = [res.projected_to_subbasis(res_range_dims[-1], dim) for
                                      res, res_range_dims in zip(self.dual_residuals,
                                                                 dual_residuals_range_dims)]
            if len(dual_residuals) == 0:
                self.logger.warning('Cannot efficiently reduce dual to subbasis')
                # the above if statements were not triggered
                dual_residuals = [res.projected_to_subbasis(None, dim)
                                  for res in self.dual_residuals]
            return CoerciveRBEstimator(residual, residual_range_dims, self.coercivity_estimator,
                                       dual_residuals, dual_residuals_range_dims,
                                       restricted_dual_models)
        else:
            self.logger.warning('Cannot efficiently reduce to subbasis')
            projected_dual_residuals = [res.projected_to_subbasis(None, dim)
                                        for res in self.dual_residuals]
            return CoerciveRBEstimator(self.residual.projected_to_subbasis(None, dim), None,
                                       self.coercivity_estimator, projected_dual_residuals,
                                       restricted_dual_models)


class SimpleCoerciveRBReductor(StationaryRBReductor):
    """Reductor for linear |StationaryModels| with affinely decomposed operator and rhs.

    .. note::
       The reductor :class:`CoerciveRBReductor` can be used for arbitrary coercive
       |StationaryModels| and offers an improved error estimator
       with better numerical stability.

    The only addition is to :class:`~pymor.reductors.basic.StationaryRBReductor` is an error
    estimator, which evaluates the norm of the residual with respect to a given inner product.

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
    """

    def __init__(self, fom, RB=None, product=None, coercivity_estimator=None,
                 check_orthonormality=None, check_tol=None,
                 assemble_error_estimate=True, assemble_output_error_estimate=True):
        assert fom.operator.linear and fom.rhs.linear
        assert isinstance(fom.operator, LincombOperator)
        assert all(not op.parametric for op in fom.operator.operators)
        if fom.rhs.parametric:
            assert isinstance(fom.rhs, LincombOperator)
            assert all(not op.parametric for op in fom.rhs.operators)

        super().__init__(fom, RB, product=product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol, assemble_error_estimate=assemble_error_estimate,
                         assemble_output_error_estimate=assemble_output_error_estimate)
        self.coercivity_estimator = coercivity_estimator
        self.extends = None

    def assemble_error_estimator(self):
        fom, RB, extends = self.fom, self.bases['RB'], self.extends
        if extends:
            old_RB_size = extends[0]
            old_data = extends[1]
        else:
            old_RB_size = 0

        # compute data for error estimator
        space = fom.operator.source

        # compute the Riesz representative of (U, .)_L2 with respect to product
        def riesz_representative(U):
            if self.products['RB'] is None:
                return U.copy()
            else:
                return self.products['RB'].apply_inverse(U)

        def append_vector(U, R, RR):
            RR.append(riesz_representative(U), remove_from_other=True)
            R.append(U, remove_from_other=True)

        # compute all components of the residual
        if extends:
            R_R, RR_R = old_data['R_R'], old_data['RR_R']
        elif not fom.rhs.parametric:
            R_R = space.empty(reserve=1)
            RR_R = space.empty(reserve=1)
            append_vector(fom.rhs.as_range_array(), R_R, RR_R)
        else:
            R_R = space.empty(reserve=len(fom.rhs.operators))
            RR_R = space.empty(reserve=len(fom.rhs.operators))
            for op in fom.rhs.operators:
                append_vector(op.as_range_array(), R_R, RR_R)

        if len(RB) == 0:
            R_Os = [space.empty()]
            RR_Os = [space.empty()]
        elif not fom.operator.parametric:
            R_Os = [space.empty(reserve=len(RB))]
            RR_Os = [space.empty(reserve=len(RB))]
            for i in range(len(RB)):
                append_vector(-fom.operator.apply(RB[i]), R_Os[0], RR_Os[0])
        else:
            R_Os = [space.empty(reserve=len(RB)) for _ in range(len(fom.operator.operators))]
            RR_Os = [space.empty(reserve=len(RB)) for _ in range(len(fom.operator.operators))]
            if old_RB_size > 0:
                for op, R_O, RR_O, old_R_O, old_RR_O in zip(fom.operator.operators, R_Os, RR_Os,
                                                            old_data['R_Os'], old_data['RR_Os']):
                    R_O.append(old_R_O)
                    RR_O.append(old_RR_O)
            for op, R_O, RR_O in zip(fom.operator.operators, R_Os, RR_Os):
                for i in range(old_RB_size, len(RB)):
                    append_vector(-op.apply(RB[i]), R_O, RR_O)

        # compute Gram matrix of the residuals
        R_RR = RR_R.inner(R_R)
        R_RO = np.hstack([RR_R.inner(R_O) for R_O in R_Os])
        R_OO = np.vstack([np.hstack([RR_O.inner(R_O) for R_O in R_Os]) for RR_O in RR_Os])

        estimator_matrix = np.empty((len(R_RR) + len(R_OO),) * 2)
        estimator_matrix[:len(R_RR), :len(R_RR)] = R_RR
        estimator_matrix[len(R_RR):, len(R_RR):] = R_OO
        estimator_matrix[:len(R_RR), len(R_RR):] = R_RO
        estimator_matrix[len(R_RR):, :len(R_RR)] = R_RO.T

        estimator_matrix = NumpyMatrixOperator(estimator_matrix)

        error_estimator = SimpleCoerciveRBEstimator(estimator_matrix, self.coercivity_estimator)
        self.extends = (len(RB), dict(R_R=R_R, RR_R=RR_R, R_Os=R_Os, RR_Os=RR_Os))

        return error_estimator

    def assemble_error_estimator_for_subbasis(self, dims):
        return self._last_rom.estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)


class SimpleCoerciveRBEstimator(ImmutableObject):
    """Instantiated by :class:`SimpleCoerciveRBReductor`.

    Not to be used directly.
    """

    def __init__(self, estimator_matrix, coercivity_estimator):
        self.__auto_init(locals())
        self.norm = induced_norm(estimator_matrix)

    def estimate_error(self, U, mu, m):
        if len(U) > 1:
            raise NotImplementedError
        if not m.rhs.parametric:
            CR = np.ones(1)
        else:
            CR = np.array(m.rhs.evaluate_coefficients(mu))

        if not m.operator.parametric:
            CO = np.ones(1)
        else:
            CO = np.array(m.operator.evaluate_coefficients(mu))

        C = np.hstack((CR, np.dot(CO[..., np.newaxis], U.to_numpy()).ravel()))

        est = self.norm(NumpyVectorSpace.make_array(C))
        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)

        return est

    def restricted_to_subbasis(self, dim, m):
        cr = 1 if not m.rhs.parametric else len(m.rhs.operators)
        co = 1 if not m.operator.parametric else len(m.operator.operators)
        old_dim = m.operator.source.dim

        indices = np.concatenate((np.arange(cr),
                                 ((np.arange(co)*old_dim)[..., np.newaxis] + np.arange(dim)).ravel() + cr))
        matrix = self.estimator_matrix.matrix[indices, :][:, indices]

        return SimpleCoerciveRBEstimator(NumpyMatrixOperator(matrix), self.coercivity_estimator)
