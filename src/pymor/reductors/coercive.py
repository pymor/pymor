# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.image import estimate_image
from pymor.algorithms.projection import project
from pymor.core.base import ImmutableObject
from pymor.operators.constructions import LincombOperator, induced_norm
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ParameterFunctional, ConstantParameterFunctional
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

    def assemble_error_estimator(self):
        residual = self.residual_reductor.reduce()

        # output estimate
        if self.fom.output_functional.linear:
            output_adjoint = self.fom.output_functional.H
            output_adjoint_riesz_range = estimate_image(vectors=(output_adjoint,), orthonormalize=True,
                                                        product=self.products['RB'], riesz_representatives=True)
            projected_output_adjoint = project(output_adjoint, output_adjoint_riesz_range, None)
        else:
            projected_output_adjoint = None

        return CoerciveRBEstimator(residual, tuple(self.residual_reductor.residual_range_dims),
                                   self.coercivity_estimator, projected_output_adjoint)

    def assemble_error_estimator_for_subbasis(self, dims):
        return self._last_rom.error_estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)


class CoerciveRBEstimator(ImmutableObject):
    """Instantiated by :class:`CoerciveRBReductor`.

    Not to be used directly.
    """

    def __init__(self, residual, residual_range_dims, coercivity_estimator, projected_output_adjoint=None):
        self.__auto_init(locals())

    def estimate_error(self, U, mu, m):
        est = self.residual.apply(U, mu=mu).norm()
        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)
        return est

    def estimate_output_error(self, U, mu, m, return_vector=False):
        if self.projected_output_adjoint is None:
            raise NotImplementedError
        estimate = self.estimate_error(U, mu, m)
        # scale with dual norm of the output functional
        output_functional_norms = self.projected_output_adjoint.as_range_array(mu).norm()
        errs = estimate * output_functional_norms
        if return_vector:
            return errs
        else:
            return np.linalg.norm(errs)

    def restricted_to_subbasis(self, dim, m):
        if self.residual_range_dims:
            residual_range_dims = self.residual_range_dims[:dim + 1]
            residual = self.residual.projected_to_subbasis(residual_range_dims[-1], dim)
            return CoerciveRBEstimator(residual, residual_range_dims, self.coercivity_estimator,
                                       self.projected_output_adjoint)
        else:
            self.logger.warning('Cannot efficiently reduce to subbasis')
            return CoerciveRBEstimator(self.residual.projected_to_subbasis(None, dim), None,
                                       self.coercivity_estimator, self.projected_output_adjoint)


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

        # output estimate
        if self.fom.output_functional.linear:
            output_estimator_matrices, output_functional_coeffs = self.assemble_output_error_estimator()
        else:
            output_estimator_matrices = output_functional_coeffs = None

        error_estimator = SimpleCoerciveRBEstimator(estimator_matrix, self.coercivity_estimator,
                                                    output_estimator_matrices, output_functional_coeffs)
        self.extends = (len(RB), dict(R_R=R_R, RR_R=RR_R, R_Os=R_Os, RR_Os=RR_Os))

        return error_estimator

    def assemble_output_error_estimator(self):
        output_func = self.fom.output_functional
        product = self.products['RB']
        if not isinstance(output_func, LincombOperator):
            output_func = LincombOperator([output_func, ], [1, ])
        assert all(not op.parametric for op in output_func.operators)
        assert all(op.linear for op in output_func.operators)
        # compute gramian of the riesz representatives
        output_estimator_matrices, output_functional_coeffs = [], []
        for d in range(self.fom.dim_output):
            riesz_representatives = [product.apply_inverse(func.as_source_array()[d])
                                     for func in output_func.operators]
            output_estimator_matrix = np.array([[product.apply2(rr, ss)[0][0] for rr in riesz_representatives]
                                                for ss in riesz_representatives])
            del riesz_representatives
            output_estimator_matrices.append(output_estimator_matrix)
        # wrap coefficient functionals if required
        output_functional_coeffs = [c if isinstance(c, ParameterFunctional) else ConstantParameterFunctional(c)
                                    for c in output_func.coefficients]
        return output_estimator_matrices, output_functional_coeffs

    def assemble_error_estimator_for_subbasis(self, dims):
        return self._last_rom.error_estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)


class SimpleCoerciveRBEstimator(ImmutableObject):
    """Instantiated by :class:`SimpleCoerciveRBReductor`.

    Not to be used directly.
    """

    def __init__(self, estimator_matrix, coercivity_estimator, output_estimator_matrices, output_functional_coeffs):
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

    def estimate_output_error(self, U, mu, m, return_vector=False):
        if not self.output_estimator_matrices or not self.output_functional_coeffs:
            raise NotImplementedError
        estimate = self.estimate_error(U, mu, m)
        # scale with dual norm of the output functional
        coeff_vals = np.array([c.evaluate(mu) for c in self.output_functional_coeffs])
        dual_norms = []
        for d in range(m.dim_output):
            dual_norms.append(np.sqrt(coeff_vals.T@(self.output_estimator_matrices[d]@coeff_vals)))
        errs = estimate * dual_norms
        if return_vector:
            return errs
        else:
            return np.linalg.norm(errs)

    def restricted_to_subbasis(self, dim, m):
        cr = 1 if not m.rhs.parametric else len(m.rhs.operators)
        co = 1 if not m.operator.parametric else len(m.operator.operators)
        old_dim = m.operator.source.dim

        indices = np.concatenate((np.arange(cr),
                                 ((np.arange(co)*old_dim)[..., np.newaxis] + np.arange(dim)).ravel() + cr))
        matrix = self.estimator_matrix.matrix[indices, :][:, indices]

        return SimpleCoerciveRBEstimator(NumpyMatrixOperator(matrix), self.coercivity_estimator,
                                         self.output_estimator_matrices, self.output_functional_coeffs)
