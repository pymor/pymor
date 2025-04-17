# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.projection import project
from pymor.algorithms.rand_la import RandomizedRangeFinder
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.cache import CacheableObject, cached
from pymor.models.iosys import LTIModel
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyHankelOperator, NumpyMatrixOperator


class ERAReductor(CacheableObject):
    r"""Eigensystem Realization Algorithm reductor.

    Constructs a (reduced) realization from a sequence of Markov parameters :math:`h_i`,
    for :math:`i\in\{1,\,\dots,\,2s-1\}`, :math:`s\in\mathbb{N}`, by a (reduced) orthogonal
    factorization of the Hankel matrix of Markov parameters

    .. math::
        H =
        \begin{bmatrix}
            h_1 & h_2 & \dots & h_s \\
            h_2 & h_3 & \dots & h_{s+1}\\
            \vdots & \vdots & \ddots & \vdots\\
            h_s & h_{s+1} & \dots & h_{2s-1}
        \end{bmatrix}=U\Sigma V^T\in\mathbb{R}^{ps\times ms},

    where :math:`r\leq\min\{ms,ps\}` is the reduced order. See :cite:`K78`.

    In order for the identified model to be stable, the Markov parameters decay substantially within
    :math:`s` samples. Stability is enforced automatically through zero-padding and can be
    deactivated by setting `force_stability=False`.

    For a large number of inputs and/or outputs, the factorization of the Hankel matrix can be
    accelerated by tangentially projecting the Markov parameters to reduce the dimension of
    the Hankel matrix, i.e.

    .. math::
        \hat{h}_i = W_L^T h_i W_R,

    where :math:`n_L \leq p` and :math:`n_R \leq m` are the number of left and right tangential
    directions and :math:`W_L \in \mathbb{R}^{p \times n_L}` an
    :math:`W_R \in \mathbb{R}^{m \times n_R}` are the left and right projectors, respectively.
    See :cite:`KG16`.

    Attributes
    ----------
    data
        |NumPy array| that contains the first :math:`n` Markov parameters of an LTI system.
        Has to be one- or three-dimensional with either::

            data.shape == (n,)

        for scalar-valued Markov parameters or::

            data.shape == (n, p, m)

        for matrix-valued Markov parameters of dimension :math:`p\times m`, where
        :math:`m` is the number of inputs and :math:`p` is the number of outputs of the system.
    sampling_time
        A number that denotes the sampling time of the system (in seconds).
    force_stability
        Whether the Markov parameters are zero-padded to double the length in order to enforce
        Kung's stability assumption. See :cite:`K78`. Defaults to `True`.
    feedthrough
        (Optional) |Operator| or |Numpy array| of shape `(p, m)`. The zeroth Markov parameter that
        defines the feedthrough of the realization. Defaults to `None`.
    """

    cache_region = 'memory'

    def __init__(self, data, sampling_time, force_stability=True, feedthrough=None):
        assert sampling_time >= 0
        assert feedthrough is None or isinstance(feedthrough, (np.ndarray, Operator))
        assert np.isrealobj(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1, 1)
        assert data.ndim == 3
        if isinstance(feedthrough, np.ndarray):
            feedthrough = NumpyMatrixOperator(feedthrough)
        if isinstance(feedthrough, Operator):
            assert feedthrough.range.dim == data.shape[1]
            assert feedthrough.source.dim == data.shape[2]
        self.__auto_init(locals())

    @cached
    def _s1_W1(self):
        self.logger.info('Computing output SVD ...')
        W1, s1, _ = spla.svd(np.hstack(self.data), full_matrices=False)
        return s1, W1

    @cached
    def _s2_W2(self):
        self.logger.info('Computing input SVD ...')
        _, s2, W2 = spla.svd(np.vstack(self.data), full_matrices=False)
        return s2, W2.T

    def _project_markov_parameters(self, num_left, num_right):
        self.logger.info('Projecting Markov parameters ...')
        mp = self.data
        # computation strategy below is already improved but probably not optimal, see PR #1587.
        if num_right:
            mp = np.einsum('npm,mk->npk', mp, self.input_projector(num_right), optimize='optimal')
        if num_left:
            mp = self.output_projector(num_left).T @ mp
        return mp

    @cached
    def _sv_U_V(self, num_left, num_right):
        n, p, m = self.data.shape
        s = n if self.force_stability else (n + 1) // 2

        h = self._project_markov_parameters(num_left, num_right) if num_left or num_right else self.data
        self.logger.info(f'Computing SVD of the {"projected " if num_left or num_right else ""}Hankel matrix ...')
        if self.force_stability:
            h = np.concatenate([h, np.zeros_like(h)[1:]], axis=0)
        H = NumpyHankelOperator(h[:s], r=h[s-1:])
        U, sv, Vh = spla.svd(to_matrix(H), full_matrices=False)
        return sv, U, Vh.T

    def output_projector(self, num_left):
        """Construct the left/output projector :math:`W_1`."""
        assert isinstance(num_left, int)
        assert num_left <= self.data.shape[1]
        self.logger.info(f'Constructing output projector ({num_left} tangential directions) ...')
        return self._s1_W1()[1][:, :num_left]

    def input_projector(self, num_right):
        """Construct the right/input projector :math:`W_2`."""
        assert isinstance(num_right, int)
        assert num_right <= self.data.shape[2]
        self.logger.info(f'Constructing input projector ({num_right} tangential directions) ...')
        return self._s2_W2()[1][:, :num_right]

    def error_bounds(self, num_left=None, num_right=None):
        r"""Compute the error bounds for all possible reduction orders.

        Without tangential projection of the Markov parameters, the :math:`\mathcal{L}_2`-error
        of the Markov parameters :math:`\epsilon` is bounded by

        .. math::
            \epsilon^2 =
            \sum_{i = 1}^{2 s - 1}
            \lVert C_r A_r^{i - 1} B_r - h_i \rVert_F^2
            \leq
            \sigma_{r + 1}(\mathcal{H})
            \sqrt{r + p + m},

        where :math:`(A_r,B_r,C_r)` is the reduced realization of order :math:`r`,
        :math:`h_i\in\mathbb{R}^{p\times m}` is the :math:`i`-th Markov parameter
        and :math:`\sigma_{r+1}(\mathcal{H})` is the first neglected singular value of the
        Hankel matrix of Markov parameters.

        With tangential projection, the bound is given by

        .. math::
            \epsilon^2 =
            \sum_{i = 1}^{2 s - 1}
            \lVert C_r A_r^{i - 1} B_r - h_i \rVert_F^2
            \leq
            4 \left(
                \sum_{i = n_L + 1}^p \sigma_i^2(\Theta_L)
                + \sum_{i = n_R + 1}^m \sigma_i^2(\Theta_R)
            \right)
            + 2 \sigma_{r + 1}(\mathcal{H}) \sqrt{r + n_L + n_R},

        where :math:`\Theta_L,\,\Theta_R` is the matrix of horizontally or vertically stacked Markov
        parameters, respectively. See :cite:`KG16` (Thm. 3.4) for details.
        """
        n, p, m = self.data.shape
        s = n if self.force_stability else (n + 1) // 2

        sv = self._sv_U_V(num_left, num_right)[0]

        a = p * s if num_right is None and p * s < m else (num_right or m)
        b = m * s if num_left is None and m * s < p else (num_left or p)
        err = ((np.arange(len(sv)) + a + b) * sv**2)[1:]

        err = 2 * err if num_left or num_right else err
        if num_left:
            s1 = self._s1_W1()[0]
            err += 4 * np.linalg.norm(s1[num_left:])**2
        if num_right:
            s2 = self._s2_W2()[0]
            err += 4 * np.linalg.norm(s2[num_right:])**2

        return np.sqrt(err)

    def _construct_realization(self, sv, U, V, m, p, num_left, num_right):
        m, p = num_right or m, num_left or p
        sqsv = np.sqrt(sv)
        A, *_ = spla.lstsq(U[: -p], U[p:])
        A = NumpyMatrixOperator((1/sqsv).reshape(-1,1)*A*sqsv.reshape(1,-1))
        B = NumpyMatrixOperator((V[:m]*sqsv.reshape(1, -1)).T)
        C = NumpyMatrixOperator(U[:p]*sqsv.reshape(1, -1))
        if num_left:
            self.logger.info('Backprojecting tangential output directions ...')
            W1 = self.output_projector(num_left)
            C = project(C, source_basis=None, range_basis=C.range.from_numpy(W1))
        if num_right:
            self.logger.info('Backprojecting tangential input directions ...')
            W2 = self.input_projector(num_right)
            B = project(B, source_basis=B.source.from_numpy(W2), range_basis=None)

        return LTIModel(A, B, C, D=self.feedthrough, sampling_time=self.sampling_time,
                        presets={'o_dense': np.diag(sv), 'c_dense': np.diag(sv), 'hsv': sv})

    def reduce(self, r=None, tol=None, num_left=None, num_right=None):
        """Construct a minimal realization.

        Parameters
        ----------
        r
            Order of the reduced model if `tol` is `None`, maximum order if `tol` is specified.
        tol
            Tolerance for the error bound.
        num_left
            Number of left (output) directions for tangential projection.
        num_right
            Number of right (input) directions for tangential projection.

        Returns
        -------
        rom
            Reduced-order |LTIModel|.
        """
        assert r is not None or tol is not None
        n, p, m = self.data.shape
        s = n if self.force_stability else (n + 1) // 2
        assert num_left is None or isinstance(num_left, int) and 0 < num_left < p
        assert num_right is None or isinstance(num_right, int) and 0 < num_right < m
        assert r is None or 0 < r <= min((num_left or p), (num_right or m)) * s

        if num_left is None and m * s < p:
            self.logger.info('Data has low rank! Accelerating computation with output tangential projections ...')
            num_left = m * s
        if num_right is None and p * s < m:
            self.logger.info('Data has low rank! Accelerating computation with input tangential projections ...')
            num_right = p * s

        sv, U, V = self._sv_U_V(num_left, num_right)

        if tol is not None:
            error_bounds = self.error_bounds(num_left=num_left, num_right=num_right)
            r_tol = np.argmax(error_bounds <= tol) + 1
            r = r_tol if r is None else min(r, r_tol)

        sv, U, V = sv[:r], U[:, :r], V[:, :r]

        self.logger.info(f'Constructing reduced realization of order {r} ...')
        return self._construct_realization(sv, U, V, m, p, num_left, num_right)


class RandomizedERAReductor(ERAReductor):
    def __init__(self, data, sampling_time, force_stability=True, feedthrough=None, allow_transpose=True, rrf_opts={},
                 num_left=None, num_right=None):
        super().__init__(data, sampling_time, force_stability=force_stability, feedthrough=feedthrough)
        self.__auto_init(locals())
        #data = data.copy()
        if num_left is not None or num_right is not None:
            self.logger.info('Computing the projected Markov parameters ...')
            data = self._project_markov_parameters(num_left, num_right)
        if self.force_stability:
            data = np.concatenate([data, np.zeros_like(data)[1:]], axis=0)
        s = (data.shape[0] + 1) // 2
        self._transpose = (data.shape[1] < data.shape[2]) if allow_transpose else False
        self._H = NumpyHankelOperator(data[:s], r=data[s-1:])
        if self._transpose:
            self.logger.info('Using transposed formulation.')
            self._H = self._H.H
        self._last_sv_U_V = None
        self._rrf = RandomizedRangeFinder(self._H, **rrf_opts)
        self._rrf._draw_samples = self._draw_samples

    @cached
    def _weighted_h2_norm(self):
        T = self.data.shape[0]
        s = int((T+1)/2)
        eta = np.ones(T)
        eta[1:s+1] *= np.arange(s) + 1
        eta[s+1:] *= np.arange(s-1)[::-1][:T-s-1] + 1
        return spla.norm(self.data*np.sqrt(eta.reshape(-1, 1, 1)))

    def _sv_U_V(self, num_left, num_right):
        return self._last_sv_U_V

    def _draw_samples(self, num):
        # faster way of computing the random samples for Hankel matrices
        self._rrf.logger.info(f'Taking {num} samples ...')
        V = np.zeros((self._H._circulant.source.dim, num))
        V[:self._H.source.dim] = self._H.source.random(num, distribution='normal').to_numpy().T
        return self._H.range.make_array(self._H._circulant._circular_matvec(V)[:, :self._H.range.dim])

    def reduce(self, r=None, tol=None):
        if tol is not None:
            tol *= self._weighted_h2_norm()
        last_basis_size = len(self._rrf.Q[-1])
        Q = self._rrf.find_range(basis_size=r, tol=tol)
        r = len(Q) if r is None else r
        if r > last_basis_size:
            self.logger.info('Projecting onto reduced space ...')
            B = self._H.apply_adjoint(Q).to_numpy()
            self.logger.info(f'Computing reduced SVD of size {B.shape[0]}x{B.shape[1]} ...')
            Ub, sv, Vh = np.linalg.svd(B, full_matrices=False)
            self.logger.info('Lifting left singular vectors ...')
            U = Q.lincomb(Ub.T)
            self._last_sv_U_V = (sv, U.to_numpy().T, Vh.T)
        else:
            self.logger.info('Smaller model order requested. Reusing last SVD.')
        sv, U, V = self._last_sv_U_V
        sv, U, V = sv[:r], U[:, :r], V[:, :r]
        if self._transpose:  # switch back, if transposed formulation was used
            U, V = V, U

        self.logger.info(f'Constructing reduced realization of order {r} ...')
        _, p, m = self.data.shape
        return self._construct_realization(sv, U, V, m, p, self.num_left, self.num_right)
