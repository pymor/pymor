# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.svd_va import qr_svd
from pymor.bindings.scipy import svd_lapack_driver
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel
from pymor.operators.constructions import IdentityOperator
from pymor.parameters.base import Mu
from pymor.reductors.basic import LTIPGReductor
from pymor.solvers.matrix_equations.equations import LyapunovEquation
from pymor.solvers.matrix_equations.utils import chol


class GenericBTReductor(BasicObject):
    """Generic Balanced Truncation reductor.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
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
        self.V = None
        self.W = None
        self._pg_reductor = None

    def _gramians(self):
        """Return low-rank factors of Gramians."""
        raise NotImplementedError

    def _hankel_svd(self):
        """Return singular values and vectors."""
        raise NotImplementedError

    def error_bounds(self):
        """Return error bounds for all possible reduced orders."""
        raise NotImplementedError

    def reduce(self, r=None, tol=None, projection='bfsr'):
        """Generic Balanced Truncation.

        Parameters
        ----------
        r
            Order of the reduced model if `tol` is `None`, maximum order if `tol` is specified.
        tol
            Tolerance for the error bound if `r` is `None`.
        projection
            Projection method used:

            - `'sr'`: square root method
            - `'bfsr'`: balancing-free square root method (default, since it avoids scaling by
              singular values and orthogonalizes the projection matrices, which might make it more
              accurate than the square root method)
            - `'biorth'`: like the balancing-free square root method, except it biorthogonalizes the
              projection matrices (using :func:`~pymor.algorithms.gram_schmidt.gram_schmidt_biorth`)

        Returns
        -------
        rom
            Reduced-order model.
        """
        assert r is not None or tol is not None
        assert r is None or 0 < r < self.fom.order
        assert projection in ('sr', 'bfsr', 'biorth')

        cf, of = self._gramians()
        sU, sv, sVh = self._hankel_svd()

        # find reduced order if tol is specified
        if tol is not None:
            error_bounds = self.error_bounds()
            r_tol = np.argmax(error_bounds <= tol) + 1
            r = r_tol if r is None else min(r, r_tol)
        if r > min(len(cf), len(of)):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.')

        # compute projection matrices
        self.V = cf.lincomb(sVh[:r].T)
        self.W = of.lincomb(sU[:, :r])
        if projection == 'sr':
            alpha = 1 / np.sqrt(sv[:r])
            self.V.scal(alpha)
            self.W.scal(alpha)
        elif projection == 'bfsr':
            gram_schmidt(self.V, atol=0, rtol=0, copy=False)
            gram_schmidt(self.W, atol=0, rtol=0, copy=False)
        elif projection == 'biorth':
            gram_schmidt_biorth(self.V, self.W, product=self.fom.E, copy=False)

        # find reduced-order model
        if self.fom.parametric:
            fom_mu = self.fom.with_(**{op: getattr(self.fom, op).assemble(mu=self.mu)
                                       for op in ['A', 'B', 'C', 'D', 'E']})
        else:
            fom_mu = self.fom
        self._pg_reductor = LTIPGReductor(fom_mu, self.W, self.V, projection in ('sr', 'biorth'))
        rom = self._pg_reductor.reduce()
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)


class BTReductor(GenericBTReductor):
    """Standard (Lyapunov) Balanced Truncation reductor.

    See Section 7.3 in :cite:`A05`.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter values|.
    """

    def _gramians(self):
        return self.fom.gramian('c_lr', mu=self.mu), self.fom.gramian('o_lr', mu=self.mu)

    def _hankel_svd(self):
        return self.fom._hankel_svd(mu=self.mu)

    def error_bounds(self):
        sv = self._hankel_svd()[1]
        return 2 * sv[:0:-1].cumsum()[::-1]


class FDBTReductor(GenericBTReductor):
    """Balanced Truncation reductor using frequency domain representation of Gramians.

    See :cite:`ZSW99`.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        if fom.sampling_time > 0:
            raise NotImplementedError
        super().__init__(fom, mu=mu)

    def _gramians(self):
        return self.fom.gramian('bs_c_lr', mu=self.mu), self.fom.gramian('bs_o_lr', mu=self.mu)

    def _hankel_svd(self):
        return self.fom._hankel_svd('bs', mu=self.mu)

    def error_bounds(self):
        """L-infinity error bounds for reduced order models."""
        sv = self._hankel_svd()[1]
        return 2 * sv[:0:-1].cumsum()[::-1]


class LQGBTReductor(GenericBTReductor):
    r"""Linear Quadratic Gaussian (LQG) Balanced Truncation reductor.

    See Section 3 in :cite:`MG91`.

    Parameters
    ----------
    fom
        The full-order (continuous- or discrete-time) |LTIModel| to reduce.
    mu
        |Parameter values|.
    """

    def _gramians(self):
        return self.fom.gramian('lqg_c_lr', mu=self.mu), self.fom.gramian('lqg_o_lr', mu=self.mu)

    def _hankel_svd(self):
        return self.fom._hankel_svd('lqg', mu=self.mu)

    def error_bounds(self):
        """Error bounds for the normalized coprime factors (continuous-time only)."""
        if self.fom.sampling_time > 0:
            raise NotImplementedError
        sv = self._hankel_svd()[1]
        return 2 * (sv[:0:-1] / np.sqrt(1 + sv[:0:-1]**2)).cumsum()[::-1]


class BRBTReductor(GenericBTReductor):
    r"""Bounded Real (BR) Balanced Truncation reductor.

    See :cite:`A05` (Section 7.5.3) and :cite:`OJ88`.

    Parameters
    ----------
    fom
        The full-order (continuous- or discrete-time) |LTIModel| to reduce.
    gamma
        Upper bound for the :math:`\mathcal{H}_\infty`-norm.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, gamma=1, mu=None):
        super().__init__(fom, mu=mu)
        self.gamma = gamma

    def _gramians(self):
        cf = self.fom.gramian(('br_c_lr', self.gamma), mu=self.mu)
        of = self.fom.gramian(('br_o_lr', self.gamma), mu=self.mu)
        return cf, of

    def _hankel_svd(self):
        return self.fom._hankel_svd(('br', self.gamma), mu=self.mu)

    def error_bounds(self):
        r""":math:`\mathcal{H}_\infty` error bounds (continuous-time only)."""
        if self.fom.sampling_time > 0:
            raise NotImplementedError
        sv = self._hankel_svd()[1]
        return 2 * sv[:0:-1].cumsum()[::-1]


class PRBTReductor(GenericBTReductor):
    r"""Positive Real (PR) Balanced Truncation reductor.

    See :cite:`DP84` and :cite:`BU22`.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        if fom.sampling_time > 0:
            raise NotImplementedError
        super().__init__(fom, mu=mu)

    def _gramians(self):
        cf = self.fom.gramian('pr_c_lr', mu=self.mu)
        of = self.fom.gramian('pr_o_lr', mu=self.mu)
        return cf, of

    def _hankel_svd(self):
        return self.fom._hankel_svd('pr', mu=self.mu)

    def error_bounds(self):
        raise NotImplementedError


class BSTReductor(GenericBTReductor):
    r"""Balanced Stochastic Truncation (BST) reductor.

    See :cite:`G88` and :cite:`GA04`.

    The feedthrough term `D` is assumed to have full rank. If the number of outputs is not
    larger than the number of inputs, the controllability Gramian is the usual Lyapunov Gramian
    and the observability Gramian is the solution of a Riccati equation. Otherwise, the roles of
    the Gramians are reversed.

    Parameters
    ----------
    fom
        The full-order continuous-time |LTIModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        if fom.sampling_time > 0:
            raise NotImplementedError
        super().__init__(fom, mu=mu)

    def _gramians(self):
        return self.fom.gramian('bst_c_lr', mu=self.mu), self.fom.gramian('bst_o_lr', mu=self.mu)

    def _hankel_svd(self):
        return self.fom._hankel_svd('bst', mu=self.mu)

    def error_bounds(self):
        r"""Relative :math:`\mathcal{H}_\infty` error bounds.

        For square systems, these are bounds for
        :math:`\lVert G^{-1} (G - G_r) \rVert_{\mathcal{H}_\infty}`.
        """
        sv = self._hankel_svd()[1]
        return ((1 + sv[:0:-1]) / (1 - sv[:0:-1])).cumprod()[::-1] - 1


class HinfBTReductor(GenericBTReductor):
    r""":math:`\mathcal{H}_\infty` Balanced Truncation reductor.

    See :cite:`MG91`.

    For `gamma` going to infinity, the method becomes LQG Balanced Truncation
    (see :class:`LQGBTReductor`).

    Parameters
    ----------
    fom
        The full-order continuous-time |LTIModel| to reduce.
    gamma
        Robustness margin :math:`\gamma > 1`.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, gamma, mu=None):
        if fom.sampling_time > 0:
            raise NotImplementedError
        assert gamma > 1
        super().__init__(fom, mu=mu)
        self.gamma = gamma

    def _gramians(self):
        cf = self.fom.gramian(('hinf_c_lr', self.gamma), mu=self.mu)
        of = self.fom.gramian(('hinf_o_lr', self.gamma), mu=self.mu)
        return cf, of

    def _hankel_svd(self):
        return self.fom._hankel_svd(('hinf', self.gamma), mu=self.mu)

    def error_bounds(self):
        """Error bounds for the normalized coprime factors."""
        sv = np.sqrt(1 - self.gamma**-2) * self._hankel_svd()[1]
        return 2 * (sv[:0:-1] / np.sqrt(1 + sv[:0:-1]**2)).cumsum()[::-1]


class GenericWeightedBTReductor(GenericBTReductor):
    """Generic Balanced Truncation reductor using weighted Gramians.

    The low-rank Gramian factors and the singular value decomposition are computed once
    and cached in the reductor.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        super().__init__(fom, mu=mu)
        self._gramian_factors = None
        self._svd = None

    def _compute_gramians(self):
        """Compute low-rank factors of the weighted Gramians."""
        raise NotImplementedError

    def _gramians(self):
        if self._gramian_factors is None:
            self._gramian_factors = self._compute_gramians()
        return self._gramian_factors

    def _hankel_svd(self):
        if self._svd is None:
            cf, of = self._gramians()
            self._svd = spla.svd(self.fom.E.apply2(of, cf, mu=self.mu), lapack_driver=svd_lapack_driver())
        return self._svd

    def error_bounds(self):
        raise NotImplementedError

    def _dense_gramians(self):
        """Return dense matrices needed for computing weighted Gramians.

        Returns
        -------
        A
            The |NumPy array| :math:`E^{-1} A`.
        E
            The |NumPy array| E or `None`.
        P
            The dense controllability Gramian.
        Q
            The dense observability Gramian of the system
            :math:`(E^{-1} A, E^{-1} B, C)`, i.e., :math:`E^T Q E`.
        """
        A, _, _, _, E = self.fom.to_matrices(format='dense', mu=self.mu)
        P = self.fom.gramian('c_dense', mu=self.mu)
        Q = self.fom.gramian('o_dense', mu=self.mu)
        if E is not None:
            A = spla.solve(E, A)
            Q = E.T @ Q @ E
        return A, E, P, Q

    def _factors_from_dense(self, E, P, Q):
        """Return low-rank factors from dense Gramians (see :meth:`_dense_gramians`)."""
        cf = chol((P + P.T) / 2)
        of = chol((Q + Q.T) / 2)
        if E is not None:
            of = spla.solve(E.T, of)
        return self.fom.A.source.from_numpy(cf), self.fom.A.source.from_numpy(of)


class FWBTReductor(GenericWeightedBTReductor):
    r"""Frequency-Weighted Balanced Truncation reductor.

    Uses the weighted Gramians from :cite:`E84` (see also :cite:`GA04`).
    For a system :math:`G`, the controllability Gramian is replaced by the corresponding
    block of the controllability Gramian of :math:`G W_i` and the observability Gramian by the
    corresponding block of the observability Gramian of :math:`W_o G`, where :math:`W_i` and
    :math:`W_o` are the input and output weights.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    input_weight
        The input weight as an |LTIModel| or `None` (no input weight).
    output_weight
        The output weight as an |LTIModel| or `None` (no output weight).
    mu
        |Parameter values|.
    """

    def __init__(self, fom, input_weight=None, output_weight=None, mu=None):
        super().__init__(fom, mu=mu)
        assert input_weight is not None or output_weight is not None
        assert (input_weight is None
                or isinstance(input_weight, LTIModel)
                and input_weight.dim_output == fom.dim_input
                and input_weight.sampling_time == fom.sampling_time)
        assert (output_weight is None
                or isinstance(output_weight, LTIModel)
                and output_weight.dim_input == fom.dim_output
                and output_weight.sampling_time == fom.sampling_time)
        self.input_weight = input_weight
        self.output_weight = output_weight

    def _compute_gramians(self):
        solvers = self.fom.matrix_equation_solvers
        if self.input_weight is None:
            cf = self.fom.gramian('c_lr', mu=self.mu)
        else:
            fom_wi = (self.fom * self.input_weight).with_(matrix_equation_solvers=solvers)
            cf = fom_wi.gramian('c_lr', mu=self.mu).blocks[0]
        if self.output_weight is None:
            of = self.fom.gramian('o_lr', mu=self.mu)
        else:
            wo_fom = (self.output_weight * self.fom).with_(matrix_equation_solvers=solvers)
            of = wo_fom.gramian('o_lr', mu=self.mu).blocks[1]
        return cf, of


class FLBTReductor(GenericWeightedBTReductor):
    r"""Frequency-Limited Balanced Truncation reductor.

    Uses the frequency-limited Gramians from :cite:`GJ90` (see also :cite:`GA04`),
    i.e., the Gramians are restricted to the frequency range
    :math:`[-\omega_2, -\omega_1] \cup [\omega_1, \omega_2]`.

    .. note::
        The Gramians are computed using dense matrices.
        Furthermore, the reduced-order model is not guaranteed to be asymptotically stable.

    Parameters
    ----------
    fom
        The full-order continuous-time |LTIModel| to reduce.
    omega
        The frequency interval as a pair :math:`(\omega_1, \omega_2)` with
        :math:`0 \le \omega_1 < \omega_2 \le \infty`.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, omega, mu=None):
        if fom.sampling_time > 0:
            raise NotImplementedError
        assert len(omega) == 2
        assert 0 <= omega[0] < omega[1]
        super().__init__(fom, mu=mu)
        self.omega = omega

    def _compute_gramians(self):
        A, E, P, Q = self._dense_gramians()
        S = _freq_limit_matrix(A, self.omega[1]) - _freq_limit_matrix(A, self.omega[0])
        P = S @ P + P @ S.T
        Q = S.T @ Q + Q @ S
        return self._factors_from_dense(E, P, Q)


class TLBTReductor(GenericWeightedBTReductor):
    r"""Time-Limited Balanced Truncation reductor.

    Uses the time-limited Gramians from :cite:`GJ90` (see also :cite:`GA04`),
    i.e., the Gramians are restricted to the time interval
    :math:`[t_1, t_2]`.

    .. note::
        The Gramians are computed using dense matrices.
        Furthermore, the reduced-order model is not guaranteed to be asymptotically stable.

    Parameters
    ----------
    fom
        The full-order continuous-time |LTIModel| to reduce.
    time_interval
        The time interval as a pair :math:`(t_1, t_2)` with
        :math:`0 \le t_1 < t_2 \le \infty`.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, time_interval, mu=None):
        if fom.sampling_time > 0:
            raise NotImplementedError
        assert len(time_interval) == 2
        assert 0 <= time_interval[0] < time_interval[1]
        super().__init__(fom, mu=mu)
        self.time_interval = time_interval

    def _compute_gramians(self):
        A, E, P, Q = self._dense_gramians()
        P_T = np.zeros_like(P)
        Q_T = np.zeros_like(Q)
        for t, sign in zip(self.time_interval, (1, -1), strict=True):
            if t == np.inf:
                continue
            F = spla.expm(t * A)
            P_T += sign * (F @ P @ F.T)
            Q_T += sign * (F.T @ Q @ F)
        return self._factors_from_dense(E, P_T, Q_T)


class TWBTReductor(GenericWeightedBTReductor):
    r"""Time-Weighted Balanced Truncation reductor.

    Uses the time-weighted Gramians from :cite:`SDM95` and :cite:`S02`, i.e.,

    .. math::
        P_k = \frac{1}{k!} \int_0^\infty t^k e^{A t} B B^T e^{A^T t} \operatorname{d}\!t, \quad
        Q_k = \frac{1}{k!} \int_0^\infty t^k e^{A^T t} C^T C e^{A t} \operatorname{d}\!t,

    which are computed by solving a sequence of :math:`k + 1` Lyapunov equations
    (:math:`A P_i + P_i A^T + P_{i - 1} = 0` and analogously for :math:`Q_i`).
    After each solve, the low-rank Gramian factor is compressed using
    :func:`~pymor.algorithms.svd_va.qr_svd`.

    Parameters
    ----------
    fom
        The full-order continuous-time |LTIModel| to reduce.
    k
        The power of the time weight :math:`t^k`.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, k=1, mu=None):
        if fom.sampling_time > 0:
            raise NotImplementedError
        assert isinstance(k, int)
        assert k >= 0
        super().__init__(fom, mu=mu)
        self.k = k

    def _compute_gramians(self):
        cf = self.fom.gramian('c_lr', mu=self.mu)
        of = self.fom.gramian('o_lr', mu=self.mu)
        A = self.fom.A.assemble(self.mu)
        E = None if isinstance(self.fom.E, IdentityOperator) else self.fom.E.assemble(self.mu)
        solver = self.fom.matrix_equation_solvers.lyapunov_lr
        for _ in range(self.k):
            cf = _compress(cf)
            of = _compress(of)
            cf = LyapunovEquation(A, E, E.apply(cf) if E is not None else cf, trans=False).solve_lr(solver=solver)
            of = LyapunovEquation(A, E, E.apply_adjoint(of) if E is not None else of, trans=True).solve_lr(
                solver=solver
            )
        return _compress(cf), _compress(of)


def _compress(Z):
    """Compress a low-rank Gramian factor using its singular value decomposition."""
    U, s, _ = qr_svd(Z)
    U.scal(s)
    return U


def _freq_limit_matrix(A, omega):
    r"""Compute the frequency-limited Gramian matrix function.

    Computes the real matrix

    .. math::
        S(\omega)
        = \frac{1}{2 \pi} \int_{-\omega}^{\omega} (\imath \nu I - A)^{-1} \operatorname{d}\!\nu
        = \operatorname{Re}\left(
            \frac{\imath}{2 \pi}
            \ln\left((A + \imath \omega I) (A - \imath \omega I)^{-1}\right)
        \right)

    for an asymptotically stable matrix `A`.
    """
    n = A.shape[0]
    if omega == 0:
        return np.zeros((n, n))
    if omega == np.inf:
        return np.eye(n) / 2
    I = np.eye(n)
    M = spla.solve(A - 1j * omega * I, A + 1j * omega * I)
    return (1j / (2 * np.pi) * spla.logm(M)).real
