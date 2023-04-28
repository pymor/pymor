# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel
from pymor.parameters.base import Mu
from pymor.algorithms.lyapunov import solve_cont_lyap_dense
from pymor.algorithms.to_matrix import to_matrix
from pymor.algorithms.lyapunov import _chol
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.h2 import IRKAReductor
from pymor.reductors.bt import BTReductor

class SpectralFactorReductor(BasicObject):
    r"""
    Passivity preserving model reduction via spectral factorization.

    See :cite:`BU22`.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, LTIModel)
        if not isinstance(mu, Mu):
            mu = fom.parameters.parse(mu)
        assert fom.parameters.assert_compatible(mu)
        self.fom = fom
        self.mu = mu

    def reduce(self, r=None):
        A = to_matrix(self.fom.A, format='dense')
        B = to_matrix(self.fom.B, format='dense')
        C = to_matrix(self.fom.C, format='dense')
        D = to_matrix(self.fom.D, format='dense')
        print('[spectralFactor] A')
        print(A)

        # Compute minimal X
        Z = self.fom.gramian('pr_o_lrcf', mu=self.mu).to_numpy()
        # TODO Add option to supply X from outside?
        # TODO Do we really need to compute the full X out of the low-rank factor Z?
        # TODO However, currently, `solve_pos_ricc_lrcf` uses a dense solver in the background anyway?
        X = Z.T@Z
        print('[spectralFactor] X')
        print(X)
        
        # Compute Cholesky-like factorization of W(X)
        M = _chol(D+D.T).T
        # TODO Alternatives to taking matrix inverse?
        L = np.linalg.solve(M.T, C-B.T@X)
        print('[spectralFactor] L')
        print(L)
        # TODO Currently, relative LTL error way too high?
        print('[spectralFactor] Relative LTL error')
        LTL = -A.T@X - X@A
        print(np.linalg.norm(L.T@L-LTL)/np.linalg.norm(LTL))
        print('[spectralFactor] M')
        print(M)
        print('[spectralFactor] Relative LTM error')
        LTM = C.T - X@B
        print(np.linalg.norm(L.T@M-LTM)/np.linalg.norm(LTM))

        spectral_factor = LTIModel(self.fom.A, self.fom.B,
            NumpyMatrixOperator(L, source_id=self.fom.A.range.id),
            NumpyMatrixOperator(M, source_id=self.fom.B.source.id))
        
        # TODO Allow to set other reductor or reductor options from outside
        irka = IRKAReductor(spectral_factor, self.mu)
        spectral_factor_reduced = irka.reduce(r)
        # bt = BTReductor(spectral_factor, self.mu)
        # spectral_factor_reduced = bt.reduce(r, projection='sr')

        spectralH2err = spectral_factor_reduced - spectral_factor
        print('[spectralFactor] Relative H2 error of spectral factor: '
            f'{spectralH2err.h2_norm() / spectral_factor.h2_norm():.3e}')

        Ar = to_matrix(spectral_factor_reduced.A, format='dense')
        Br = to_matrix(spectral_factor_reduced.B, format='dense')
        Lr = to_matrix(spectral_factor_reduced.C, format='dense')
        Mr = to_matrix(spectral_factor_reduced.D, format='dense')

        ev = np.max(np.real(np.linalg.eig(Ar)[0]))
        if ev > 0:
            print(f'[spectralFactor] Warning: reduced spectral factor not stable. {ev}')

        print('[spectralFactor] Mr')
        print(Mr)

        Dr = 0.5*(Mr.T @ Mr) + 0.5*(D-D.T)

        Xr = solve_cont_lyap_dense(A=Ar, E=None, B=Lr, trans=True)
        Cr = Br.T @ Xr + Mr.T @ Lr

        return LTIModel(spectral_factor_reduced.A, spectral_factor_reduced.B,
            NumpyMatrixOperator(Cr, source_id=spectral_factor_reduced.A.range.id),
            NumpyMatrixOperator(Dr, source_id=spectral_factor_reduced.B.source.id))