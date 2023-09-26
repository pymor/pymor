# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.lyapunov import _chol, solve_cont_lyap_dense
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel
from pymor.operators.constructions import ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Mu


class SpectralFactorReductor(BasicObject):
    r"""Passivity preserving model reduction via spectral factorization.

    See :cite:`BU22` (Algorithm 4).

    .. note::
        The reductor uses dense computations and converts the full-order model to dense matrices.

    Parameters
    ----------
    fom
        The passive full-order |LTIModel| to reduce. The full-order model must be
        minimal and asymptotically stable.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, LTIModel)
        if not isinstance(mu, Mu):
            mu = fom.parameters.parse(mu)
        assert fom.parameters.assert_compatible(mu)
        self.fom = fom
        self.mu = mu

    def reduce(self, r_fn, X=None, compute_errors=False, check_stability=True):
        r"""Reduce system by reducing its spectral factor.

        Parameters
        ----------
        r_fn
            A callable which takes two arguments

            - `spectral_factor` (|LTIModel|),
            - `mu` (|Parameter values|),

            and returns a reduced-order |LTIModel| for the supplied spectral
            factor.

            For example, a possible choice to obtain a reduced-order model of order 10 is::

                lambda spectral_factor, mu: IRKAReductor(spectral_factor, mu).reduce(10)

            or::

                lambda spectral_factor, mu: BTReductor(spectral_factor, mu).reduce(10)

            The method should preserve asymptotic stability.
        X
            A solution to the KYP inequality

            .. math::
                \begin{bmatrix}
                    -A^T X - X A & C^T - X B \\
                    C - B^T X & D + D^T
                \end{bmatrix}
                \succcurlyeq 0.

            as a |NumPy array|, which in turn is used for computation of the spectral
            factor.

            If `None`, it is assumed that :math:`D + D^T` is nonsingular, where
            :math:`D` is the feed-through matrix of the full-order model.
            A minimal solution to the KYP inequality is then obtained internally
            by computing the minimal solution of the Riccati equation

            .. math::
                A^T X E + E^T X A
                + (C^T - E^T X B) (D + D^T)^{-1} (C - B^T X E) = 0.

            In the case that :math:`D+D^T` is singular, one can add a small
            perturbation, see :cite:`BU22` (Section 2.2.2).
        compute_errors
            If `True`, the relative :math:`\mathcal{H}_2` error of the
            reduced spectral factor is computed.
        check_stability
            If `True`, the stability of the reduced spectral factor is
            checked. The stability is required to guarantee a positive definite
            solution to the Lyapunov equation in :cite:`BU22` in equation (21).

        Returns
        -------
        rom
            Reduced passive |LTIModel| model.
        """
        if X is None:
            # Compute minimal solution X to Riccati equation
            assert not isinstance(self.fom.D, ZeroOperator), 'D+D^T must be invertible.'
            Z = self.fom.gramian('pr_o_lrcf', mu=self.mu).to_numpy()
            X = Z.T @ Z

        # Compute Cholesky-like factorization of W(X)
        E = to_matrix(self.fom.E, format='dense')
        B = to_matrix(self.fom.B, format='dense')
        C = to_matrix(self.fom.C, format='dense')
        D = to_matrix(self.fom.D, format='dense')
        M = _chol(D + D.T).T
        L = spla.solve(M.T, C - B.T @ X @ E)

        if compute_errors:
            A = to_matrix(self.fom.A, format='dense')
            LTL = -A.T @ X @ E - X @ A @ E
            relLTLerr = np.linalg.norm(L.T @ L - LTL) / np.linalg.norm(LTL)
            self.logger.info(f'Relative L^T*L error: {relLTLerr:.3e}')
            LTM = C.T - E.T @ X @ B
            relLTMerr = np.linalg.norm(L.T @ M - LTM) / np.linalg.norm(LTM)
            self.logger.info(f'Relative L^T*M error: {relLTMerr:.3e}')

        spectral_factor = LTIModel(self.fom.A, self.fom.B,
            NumpyMatrixOperator(L, source_id=self.fom.A.range.id),
            NumpyMatrixOperator(M, source_id=self.fom.B.source.id),
            self.fom.E)

        spectral_factor_reduced = r_fn(spectral_factor, self.mu)

        if compute_errors:
            spectralH2err = spectral_factor_reduced - spectral_factor
            self.logger.info('Relative H2 error of reduced spectral factor: '
                f'{spectralH2err.h2_norm() / spectral_factor.h2_norm():.3e}')

        Er = to_matrix(spectral_factor_reduced.E, format='dense')
        Ar = to_matrix(spectral_factor_reduced.A, format='dense')
        Br = to_matrix(spectral_factor_reduced.B, format='dense')
        Lr = to_matrix(spectral_factor_reduced.C, format='dense')
        Mr = to_matrix(spectral_factor_reduced.D, format='dense')

        if check_stability:
            largest_pole = spectral_factor_reduced.poles().real.max()
            if largest_pole > 0:
                self.logger.warn('Reduced system for spectral factor is not stable. '
                                 f'Real value of largest pole is {largest_pole}.')

        Dr = 0.5*(Mr.T @ Mr) + 0.5*(D-D.T)

        Xr = solve_cont_lyap_dense(A=Ar, E=Er, B=Lr, trans=True)
        Cr = Br.T @ Xr @ Er + Mr.T @ Lr

        return LTIModel.from_matrices(Ar, Br, Cr, Dr, Er)
