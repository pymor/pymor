# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.to_matrix import to_matrix
from pymor.core.interfaces import BasicInterface
from pymor.core.logger import getLogger
from pymor.discretizations.iosys import LTISystem
from pymor.operators.constructions import IdentityOperator


class TFInterpReductor(BasicInterface):
    """Loewner bitangential Hermite interpolation reductor.

    Parameters
    ----------
    d
        Discretization with `eval_tf` and `eval_dtf` methods.
    """
    def __init__(self, d):
        self.d = d

    def reduce(self, sigma, b, c):
        """Realization-independent tangential Hermite interpolation.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of
            length `r`.
        b
            Right tangential directions, |NumPy array| of shape
            `(d.m, r)`.
        c
            Left tangential directions, |NumPy array| of shape
            `(d.p, r)`.

        Returns
        -------
        lti
            |LTISystem| interpolating the transfer function of `d`.
        """
        d = self.d
        r = len(sigma)
        assert isinstance(b, np.ndarray) and b.shape == (d.m, r)
        assert isinstance(c, np.ndarray) and c.shape == (d.p, r)

        # rescale tangential directions (to avoid overflow or underflow)
        for i in range(r):
            b[:, i] /= spla.norm(b[:, i])
            c[:, i] /= spla.norm(c[:, i])

        # matrices of the interpolatory LTI system
        Er = np.empty((r, r), dtype=complex)
        Ar = np.empty((r, r), dtype=complex)
        Br = np.empty((r, d.m), dtype=complex)
        Cr = np.empty((d.p, r), dtype=complex)

        Hs = [d.eval_tf(s) for s in sigma]
        dHs = [d.eval_dtf(s) for s in sigma]

        for i in range(r):
            for j in range(r):
                if i != j:
                    Er[i, j] = -c[:, i].dot((Hs[i] - Hs[j]).dot(b[:, j])) / (sigma[i] - sigma[j])
                    Ar[i, j] = -c[:, i].dot((sigma[i] * Hs[i] - sigma[j] * Hs[j])).dot(b[:, j]) / (sigma[i] - sigma[j])
                else:
                    Er[i, i] = -c[:, i].dot(dHs[i].dot(b[:, i]))
                    Ar[i, i] = -c[:, i].dot((Hs[i] + sigma[i] * dHs[i]).dot(b[:, i]))
            Br[i, :] = Hs[i].T.dot(c[:, i])
            Cr[:, i] = Hs[i].dot(b[:, i])

        # transform the system to have real matrices
        T = np.zeros((r, r), dtype=complex)
        for i in range(r):
            if sigma[i].imag == 0:
                T[i, i] = 1
            else:
                try:
                    j = i + 1 + np.where(np.isclose(sigma[i + 1:], sigma[i].conjugate()))[0][0]
                except:
                    j = None
                if j:
                    T[i, i] = 1
                    T[i, j] = 1
                    T[j, i] = -1j
                    T[j, j] = 1j
        Er = (T.dot(Er).dot(T.conj().T)).real
        Ar = (T.dot(Ar).dot(T.conj().T)).real
        Br = (T.dot(Br)).real
        Cr = (Cr.dot(T.conj().T)).real

        return LTISystem.from_matrices(Ar, Br, Cr, D=None, E=Er, cont_time=d.cont_time)


class TF_IRKAReductor(BasicInterface):
    """Realization-independent IRKA reductor.

    .. [AG12] C. A. Beattie, S. Gugercin, Realization-independent
              H2-approximation,
              Proceedings of the 51st IEEE Conference on Decision and
              Control, 2012.

    Parameters
    ----------
    d
        Discretization with `eval_tf` and `eval_dtf` methods.
    """
    def __init__(self, d):
        self.d = d

    def reduce(self, r, sigma=None, b=None, c=None, tol=1e-4, maxit=100, force_sigma_in_rhp=False,
               conv_crit='rel_sigma_change'):
        """Reduce using TF-IRKA.

        Parameters
        ----------
        r
            Order of the reduced order model.
        sigma
            Initial interpolation points (closed under conjugation),
            list of length `r`.

            If `None`, interpolation points are log-spaced between 0.1
            and 10.
        b
            Initial right tangential directions, |NumPy array| of shape
            `(d.m, r)`.

            If `None`, `b` is chosen with all ones.
        c
            Initial left tangential directions, |NumPy array| of shape
            `(d.p, r)`.

            If `None`, `c` is chosen with all ones.
        tol
            Tolerance for the largest change in interpolation points.
        maxit
            Maximum number of iterations.
        force_sigma_in_rhp
            If 'False`, new interpolation are reflections of reduced
            order model's poles. Otherwise, they are always in the right
            half-plane.
        conv_crit
            Convergence criterion:

                - `'rel_sigma_change'`: relative change in interpolation
                  points
                - `'rel_H2_dist'`: relative H_2 distance of reduced
                  order models

        Returns
        -------
        rd
            Reduced |LTISystem| model.
        """
        d = self.d
        if not d.cont_time:
            raise NotImplementedError
        assert r > 0
        assert sigma is None or len(sigma) == r
        assert b is None or isinstance(b, np.ndarray) and b.shape == (d.m, r)
        assert c is None or isinstance(c, np.ndarray) and c.shape == (d.p, r)
        assert conv_crit in ('rel_sigma_change', 'rel_H2_dist')

        logger = getLogger('pymor.reductors.tf.TF_IRKAReductor.reduce')
        logger.info('Starting TF-IRKA')

        # basic choice for initial interpolation points and tangential
        # directions
        if sigma is None:
            sigma = np.logspace(-1, 1, r)
        if b is None:
            b = np.ones((d.m, r))
        if c is None:
            c = np.ones((d.p, r))

        logger.info('iter | conv. criterion')
        logger.info('-----+----------------')

        self.dist = []
        self.sigmas = [np.array(sigma)]
        self.R = [b]
        self.L = [c]
        interp_reductor = TFInterpReductor(d)
        # main loop
        for it in range(maxit):
            # interpolatory reduced order model
            rd = interp_reductor.reduce(sigma, b, c)

            # new interpolation points
            if isinstance(rd.E, IdentityOperator):
                sigma, Y, X = spla.eig(to_matrix(rd.A), left=True, right=True)
            else:
                sigma, Y, X = spla.eig(to_matrix(rd.A), to_matrix(rd.E), left=True, right=True)
            if force_sigma_in_rhp:
                sigma = np.array([np.abs(s.real) + s.imag * 1j for s in sigma])
            else:
                sigma *= -1
            self.sigmas.append(sigma)

            # compute convergence criterion
            if conv_crit == 'rel_sigma_change':
                self.dist.append(spla.norm((self.sigmas[-2] - self.sigmas[-1]) / self.sigmas[-2], ord=np.inf))
            elif conv_crit == 'rel_H2_dist':
                if it == 0:
                    rd_new = rd
                    self.dist.append(np.inf)
                else:
                    rd_old = rd_new
                    rd_new = rd
                    rd_diff = rd_old - rd_new
                    try:
                        rel_H2_dist = rd_diff.norm() / rd_old.norm()
                    except:
                        rel_H2_dist = np.inf
                    self.dist.append(rel_H2_dist)

            logger.info('{:4d} | {:15.9e}'.format(it + 1, self.dist[-1]))

            # new tangential directions
            b = rd.B._matrix.T.dot(Y.conj())
            c = rd.C._matrix.dot(X)
            self.R.append(b)
            self.L.append(c)

            # check if convergence criterion is satisfied
            if self.dist[-1] < tol:
                break

        # final reduced order model
        rd = interp_reductor.reduce(sigma, b, c)

        return rd
