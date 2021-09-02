# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.base import ImmutableObject
from pymor.operators.block import BlockRowOperator
from pymor.operators.constructions import LincombOperator, induced_norm, ComponentProjectionOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import StationaryRBReductor
from pymor.reductors.residual import ResidualReductor
from pymor.vectorarrays.numpy import NumpyVectorSpace


class CoerciveRBReductor(StationaryRBReductor):
    """Reduced Basis reductor for |StationaryModels| with coercive linear operator.

    The only addition to :class:`~pymor.reductors.basic.StationaryRBReductor` is an error
    estimator which evaluates the dual norm of the residual with respect to a given inner
    product. For the reduction of the residual we use
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
    """

    def __init__(self, fom, RB=None, product=None, coercivity_estimator=None,
                 check_orthonormality=None, check_tol=None):
        super().__init__(fom, RB, product=product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol)
        self.coercivity_estimator = coercivity_estimator
        self.residual_reductor = ResidualReductor(self.bases['RB'], self.fom.operator, self.fom.rhs,
                                                  product=product, riesz_representatives=True)
        if self.fom.output_functional is not None:
            if self.fom.output_functional.linear:
                self.dual_residual_reductors = []
                for d in range(fom.output_functional.range.dim):
                    output = ComponentProjectionOperator([d], self.fom.output_functional.range) @ \
                        self.fom.output_functional
                    self.dual_residual_reductors.append(ResidualReductor(self.bases['RB'],
                                                                         self.fom.operator.H,
                                                                         output.H, product=product,
                                                                         riesz_representatives=True))

    def assemble_error_estimator(self):
        residual = self.residual_reductor.reduce()
        dual_residuals, dual_range_dims = [], []
        if self.fom.output_functional is not None:
            if self.fom.output_functional.linear:
                for dual_residual_reductor in self.dual_residual_reductors:
                    dual_residuals.append(dual_residual_reductor.reduce())
                    dual_range_dims.append(tuple(dual_residual_reductor.residual_range_dims))
        # go back to None if nothing happened (this can also happen if the for loop started)
        dual_range_dims = None if len(dual_range_dims) == 0 else dual_range_dims
        error_estimator = CoerciveRBEstimator(residual, tuple(self.residual_reductor.residual_range_dims),
                                              self.coercivity_estimator,
                                              dual_residuals, dual_range_dims)
        return error_estimator

    def assemble_error_estimator_for_subbasis(self, dims):
        return self._last_rom.error_estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)


class CoerciveRBEstimator(ImmutableObject):
    """Instantiated by :class:`CoerciveRBReductor`.

    Not to be used directly.
    """

    def __init__(self, residual, residual_range_dims, coercivity_estimator,
                 dual_residuals=None, dual_residuals_range_dims=None):
        self.__auto_init(locals())

    def estimate_error(self, U, mu, m):
        est = self.residual.apply(U, mu=mu).norm()
        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)
        return est

    def estimate_output_error(self, U, mu, m):
        assert m.output_functional is not None
        assert m.output_functional.linear
        assert len(self.dual_residuals) == m.output_functional.range.dim
        est_pr = self.estimate_error(U, mu, m)
        est_dus = []
        for d in range(m.output_functional.range.dim):
            dual_problem = m.with_(operator=m.operator.H, rhs=m.output_functional.H.as_range_array(mu)[d])
            dual_solution = dual_problem.solve(mu)
            est_dus.append(self.dual_residuals[d].apply(dual_solution, mu=mu).norm())
        return (est_pr * est_dus).T

    def restricted_to_subbasis(self, dim, m):
        if self.residual_range_dims:
            residual_range_dims = self.residual_range_dims[:dim + 1]
            residual = self.residual.projected_to_subbasis(residual_range_dims[-1], dim)
            dual_residuals = []
            dual_residuals_range_dims = []
            if self.dual_residuals_range_dims is not None:
                if len(self.dual_residuals_range_dims[0]) > 0:
                    assert len(self.dual_residuals_range_dims) == m.output_functional.range.dim
                    assert len(self.dual_residuals) == m.output_functional.range.dim
                    dual_residuals_range_dims = [res_range_dims[:dim + 1] for res_range_dims in
                                                 self.dual_residuals_range_dims]
                    dual_residuals = [res.projected_to_subbasis(res_range_dims[-1], dim) for
                                      res, res_range_dims in zip(self.dual_residuals,
                                                                 dual_residuals_range_dims)]
            if len(dual_residuals) == 0:
                # the above if statements were not triggered
                dual_residuals = [res.projected_to_subbasis(None, dim)
                                  for res in self.dual_residuals]
            return CoerciveRBEstimator(residual, residual_range_dims, self.coercivity_estimator,
                                       dual_residuals, dual_residuals_range_dims)
        else:
            self.logger.warning('Cannot efficiently reduce to subbasis')
            projected_dual_residuals = [res.projected_to_subbasis(None, dim)
                                        for res in self.dual_residuals]
            return CoerciveRBEstimator(self.residual.projected_to_subbasis(None, dim), None,
                                       self.coercivity_estimator, projected_dual_residuals)


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
                 check_orthonormality=None, check_tol=None):
        assert fom.operator.linear and fom.rhs.linear
        assert isinstance(fom.operator, LincombOperator)
        assert all(not op.parametric for op in fom.operator.operators)
        if fom.rhs.parametric:
            assert isinstance(fom.rhs, LincombOperator)
            assert all(not op.parametric for op in fom.rhs.operators)

        super().__init__(fom, RB, product=product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol)
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

        # right hand side of the dual problem if needed
        dual_estimator_matrices, R_DRs, RR_DRs = [], None, None
        if self.fom.output_functional is not None:
            if self.fom.output_functional.linear:
                if extends:
                    R_DRs, RR_DRs = old_data['R_DRs'], old_data['RR_DRs']
                elif not self.fom.output_functional.parametric:
                    R_DRs, RR_DRs = [], []
                    for d in range(self.fom.output_functional.range.dim):
                        dual_rhs = self.fom.output_functional.H
                        R_DRs.append(space.empty(reserve=1))
                        RR_DRs.append(space.empty(reserve=1))
                        append_vector(dual_rhs.as_range_array()[d], R_DRs[d], RR_DRs[d])
                else:
                    R_DRs, RR_DRs = [], []
                    for d in range(self.fom.output_functional.range.dim):
                        dual_rhs = self.fom.output_functional.H
                        if not isinstance(dual_rhs, LincombOperator):
                            # this case happens if the builtin discretizer is used for the multi-dim
                            # output, which then means that the output_functional is a BlockOperator
                            assert isinstance(dual_rhs, BlockRowOperator)
                            dual_rhs = dual_rhs.blocks[0, d]
                            R_DRs.append(space.empty(reserve=len(dual_rhs.operators)))
                            RR_DRs.append(space.empty(reserve=len(dual_rhs.operators)))
                            for op in dual_rhs.operators:
                                append_vector(op.as_range_array(), R_DRs[d], RR_DRs[d])
                        else:
                            R_DRs.append(space.empty(reserve=len(dual_rhs.operators)))
                            RR_DRs.append(space.empty(reserve=len(dual_rhs.operators)))
                            for op in dual_rhs.operators:
                                append_vector(op.as_range_array()[d], R_DRs[d], RR_DRs[d])

                # compute Gram matrix of the residuals related to dual rhs
                R_DRRs = [RR_DR.inner(R_DR) for RR_DR, R_DR in zip(RR_DRs, R_DRs)]
                R_DROs = [np.hstack([RR_DR.inner(R_O) for R_O in R_Os]) for RR_DR in RR_DRs]

                for R_DRR, R_DRO in zip(R_DRRs, R_DROs):
                    new_estimator_matrix = np.empty((len(R_DRR) + len(R_OO),) * 2)
                    new_estimator_matrix[:len(R_DRR), :len(R_DRR)] = R_DRR
                    new_estimator_matrix[len(R_DRR):, len(R_DRR):] = R_OO
                    new_estimator_matrix[:len(R_DRR), len(R_DRR):] = R_DRO
                    new_estimator_matrix[len(R_DRR):, :len(R_DRR)] = R_DRO.T

                    dual_estimator_matrices.append(NumpyMatrixOperator(new_estimator_matrix))

        error_estimator = SimpleCoerciveRBEstimator(estimator_matrix, self.coercivity_estimator,
                                                    dual_estimator_matrices)
        self.extends = (len(RB), dict(R_R=R_R, RR_R=RR_R, R_Os=R_Os, RR_Os=RR_Os,
                                      R_DRs=R_DRs, RR_DRs=RR_DRs))

        return error_estimator

    def assemble_error_estimator_for_subbasis(self, dims):
        return self._last_rom.error_estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)


class SimpleCoerciveRBEstimator(ImmutableObject):
    """Instantiated by :class:`SimpleCoerciveRBReductor`.

    Not to be used directly.
    """

    def __init__(self, estimator_matrix, coercivity_estimator, dual_estimator_matrices=None):
        self.__auto_init(locals())
        self.norm = induced_norm(estimator_matrix)
        if dual_estimator_matrices is not None:
            self.dual_norms = [induced_norm(dual_mat) for dual_mat in dual_estimator_matrices]

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

    def estimate_output_error(self, U, mu, m):
        assert m.output_functional is not None
        assert m.output_functional.linear
        if len(U) > 1:
            raise NotImplementedError
        est_pr = self.estimate_error(U, mu, m)
        est_dus = []
        for d in range(m.output_functional.range.dim):
            dual_problem = m.with_(operator=m.operator.H, rhs=m.output_functional.H.as_range_array(mu)[d])
            dual_solution = dual_problem.solve(mu)
            if not m.output_functional.parametric or not isinstance(m.output_functional, LincombOperator):
                CR = np.ones(1)
            else:
                CR = np.array(m.output_functional.evaluate_coefficients(mu))

            if not m.operator.parametric:
                CO = np.ones(1)
            else:
                CO = np.array(m.operator.evaluate_coefficients(mu))

            C = np.hstack((CR, np.dot(CO[..., np.newaxis], dual_solution.to_numpy()).ravel()))

            est = self.dual_norms[d](NumpyVectorSpace.make_array(C))
            est_dus.append(est)
        return (est_pr * est_dus).T


    def restricted_to_subbasis(self, dim, m):
        cr = 1 if not m.rhs.parametric else len(m.rhs.operators)
        co = 1 if not m.operator.parametric else len(m.operator.operators)
        old_dim = m.operator.source.dim

        indices = np.concatenate((np.arange(cr),
                                 ((np.arange(co)*old_dim)[..., np.newaxis] + np.arange(dim)).ravel() + cr))
        matrix = self.estimator_matrix.matrix[indices, :][:, indices]

        dual_estimator_matrices = []
        if m.output_functional is not None:
            if m.output_functional.linear:
                if not m.output_functional.parametric:
                    cdr = [1 for d in range(m.output_functional.range.dim)]
                elif isinstance(m.output_functional, LincombOperator):
                    cdr = [len(m.output_functional.operators) for d in range(m.output_functional.range.dim)]
                else:
                    assert isinstance(m.output_functional.H, BlockRowOperator)
                    cdr = []
                    for d in range(m.output_functional.range.dim):
                        cdr.append(len(m.output_functional.H.blocks[0, d].operators))
                for d in range(m.output_functional.range.dim):
                    indices = np.concatenate((np.arange(cdr[d]),
                                             ((np.arange(co)*old_dim)[..., np.newaxis] + np.arange(dim)).ravel()
                                              + cdr[d]))
                    new_matrix = NumpyMatrixOperator(self.dual_estimator_matrices[d].matrix[indices, :][:, indices])
                    dual_estimator_matrices.append(new_matrix)

        return SimpleCoerciveRBEstimator(NumpyMatrixOperator(matrix), self.coercivity_estimator,
                                         dual_estimator_matrices)
