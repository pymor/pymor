# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.projection import project
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.cache import cached, CacheableObject
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
            \vdots & \vdots && \vdots\\
            h_s & h_{s+1} & \dots & h_{2s-1}
        \end{bmatrix}=U\Sigma V^T\in\mathbb{R}^{ms\times ps},

    where :math:`r\leq\min\{ms,ps\}` is the reduced order. See :cite:`K78`.

    In order for the identified model to be stable, the Markov parameters must satisfy

    .. math::
        h_i\rightarrow0~\text{for}~i>s.

    Stability is enforced automatically and can be deactivated by setting `force_stability=False`.

    For a large number if inputs and/or outputs, the factorization of the Hankel matrix can be
    accelerated by tangentially projecting of the Markov parameters to reduce the dimension of
    the Hankel matrix, i.e.

    .. math::
        \hat{h}_i=W_1^Th_iW_2,

    where :math:`l_1\leq p` and :math:`l_2\leq m` are the number of left and right tangential
    directions :math:`W_1\in\mathbb{R}^{p\times l_1}` and :math:`W_2\in\mathbb{R}^{m\times l_2}`
    are the left and right projectors, respectively. See :cite:`KG16`.

    Attributes
    ----------
    data
        |NumPy array| that contains the first :math:`n` Markov parameters of a discrete-time system.
        Has to be one- or three-dimensional with either::

            data.shape == (n,)

        for scalar-valued Markov parameters or::

            data.shape == (n, p, m)

        for matrix-valued Markov parameters of dimension :math:`p\times m`, where
        :math:`m` is the number of inputs and :math:`p` is the number of outputs of the system.
    sampling_time
        A positive number that denotes the sampling time of the system (in seconds).
    force_stability
        Whether the Markov parameters are zero-padded to double the length in order to enforce
        Kung's stability assumption. See :cite:`K78`. Defaults to `True`.
    feedthrough
        (Optional) |NumpyArray| of shape `(p, m)`. The zeroth Markov parameter that defines the
        feedthrough of the realization. Defaults to `None`.
    """

    cache_region = 'memory'

    def __init__(self, data, sampling_time, force_stability=True, feedthrough=None):
        assert sampling_time >= 0
        assert feedthrough is None or isinstance(feedthrough, Operator)
        assert np.isrealobj(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1, 1)
        assert data.ndim == 3
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

    def _project_markov_parameters(self, l1, l2):
        mats = [self.output_projector(l1).T, self.data] if l1 else [self.data]
        mats = [*mats, self.input_projector(l2)] if l2 else mats
        s1 = ('lp,', 'l') if l1 else ('', 'p')
        s2 = (',mk', 'k') if l2 else ('', 'm')
        self.logger.info('Projecting Markov parameters ...')
        einstr = s1[0] + 'npm' + s2[0] + '->n' + s1[1] + s2[1]
        return np.einsum(einstr, *mats, optimize='optimal')

    @cached
    def _sv_U_V(self, l1, l2):
        n, p, m = self.data.shape
        s = n if self.force_stability else (n + 1) // 2
        if l1 is None and m * s < p:
            self.logger.info('Data has low rank! Accelerating computation with output tangential projections ...')
            l1 = m * s
        if l2 is None and p * s < m:
            self.logger.info('Data has low rank! Accelerating computation with output tangential projections ...')
            l2 = p * s
        h = self._project_markov_parameters(l1, l2) if l1 or l2 else self.data
        self.logger.info(f'Computing SVD of the {"projected " if l1 or l2 else ""}Hankel matrix ...')
        if self.force_stability:
            h = np.concatenate([h, np.zeros_like(h)[1:]], axis=0)
        U, sv, V = spla.svd(to_matrix(NumpyHankelOperator(h)), full_matrices=False)
        return sv, U.T, V

    def output_projector(self, l1):
        """Construct the left/output projector :math:`W_1`."""
        assert isinstance(l1, int) and l1 <= self.data.shape[1]
        self.logger.info(f'Constructing output projector ({l1} tangential directions) ...')
        return self._s1_W1()[1][:, :l1]

    def input_projector(self, l2):
        """Construct the right/input projector :math:`W_2`."""
        assert isinstance(l2, int) and l2 <= self.data.shape[2]
        self.logger.info(f'Constructing input projector ({l2} tangential directions) ...')
        return self._s2_W2()[1][:, :l2]

    def error_bounds(self, l1=None, l2=None):
        r"""Compute the error bounds for all possible reduction orders.

        Without tangential projection of the Markov parameters, the error bounds are defined as

        .. math::
            \sum_{i=1}^{2s-1}\lVert C_rA_r^{i-1}B_r-h_i\rVert_F^2\leq\sigma_{r+1}(\mathcal{H})
            \sqrt{r+p+m},

        where :math:`(A_r,B_r,C_r)` is the reduced realization of order :math:`r`,
        :math:`h_i\in\mathbb{R}^{p\times m}` is the :math:`i`-th Markov parameter
        and :math:`\sigma_{r+1}(\mathcal{H})` is the first neglected singular value of the
        Hankel matrix of Markov parameters.

        With tangential projection, the bound is given by

        .. math::
            \sum_{i=1}^{2s-1}\lVert C_rA_r^{i-1}B_r-h_i\rVert_F^2\leq
            4\left(\sum_{i=l_1+1}^p\sigma_i^2(\Theta_L)+\sum_{i=l_2+1}^m\sigma_i^2(\Theta_R)\right)
            +2\sigma_{r+1}(\mathcal{H})\sqrt{r+l_1+l_2},

        where :math:`\Theta_L,\,\Theta_R` is the matrix of horizontally or vertically stacked Markov
        parameters, respectively. See :cite:`KG16` (Thm. 3.4) for details.
        """
        n, p, m = self.data.shape
        s = n if self.force_stability else (n + 1) // 2

        sv = self._sv_U_V(l1, l2)[0]

        a = p * s if l2 is None and p * s < m else (l2 or m)
        b = m * s if l1 is None and m * s < p else (l1 or p)
        err = (np.sqrt(np.arange(len(sv)) + a + b) * sv)[1:]

        err = 2 * err if l1 or l2 else err
        if l1:
            s1 = self._s1_W1()[0]
            err += 4 * np.linalg.norm(s1[l1:])**2
        if l2:
            s2 = self._s2_W2()[0]
            err += 4 * np.linalg.norm(s2[l2:])**2

        return err

    def reduce(self, r=None, tol=None, l1=None, l2=None):
        """Construct a minimal realization.

        Parameters
        ----------
        r
            Order of the reduced model if `tol` is `None`, maximum order if `tol` is specified.
        tol
            Tolerance for the error bound if `r` is `None`.
        l1
            Number of left (output) directions for tangential projection.
        l2
            Number of right (input) directions for tangential projection.

        Returns
        -------
        rom
            Reduced-order discrete-time |LTIModel|.
        """
        assert r is not None or tol is not None
        n, p, m = self.data.shape
        s = n if self.force_stability else (n + 1) // 2
        assert l1 is None or isinstance(l1, int) and l1 <= p
        assert l2 is None or isinstance(l2, int) and l2 <= m
        assert r is None or 0 < r <= min((l1 or p), (l2 or m)) * s

        sv, U, V = self._sv_U_V(l1, l2)

        if tol is not None:
            error_bounds = self.error_bounds(l1=l1, l2=l2)
            r_tol = np.argmax(error_bounds <= tol) + 1
            r = r_tol if r is None else min(r, r_tol)

        sv, U, V = sv[:r], U[:r], V[:r]

        l1 = m * s if l1 is None and m * s < p else l1
        l2 = p * s if l2 is None and p * s < m else l2

        self.logger.info(f'Constructing reduced realization of order {r} ...')
        sqS = np.diag(np.sqrt(sv))
        Zo = U.T @ sqS
        A = NumpyMatrixOperator(spla.lstsq(Zo[: -(l1 or p)], Zo[(l1 or p):])[0])
        B = NumpyMatrixOperator(sqS @ V[:, :(l2 or m)])
        C = NumpyMatrixOperator(Zo[:(l1 or p)])

        if l1:
            self.logger.info('Backprojecting tangential output directions ...')
            W1 = self.output_projector(l1)
            C = project(C, source_basis=None, range_basis=C.range.from_numpy(W1))
        if l2:
            self.logger.info('Backprojecting tangential input directions ...')
            W2 = self.input_projector(l2)
            B = project(B, source_basis=B.source.from_numpy(W2), range_basis=None)

        return LTIModel(A, B, C, D=self.feedthrough, sampling_time=self.sampling_time,
                        presets={'o_dense': np.diag(sv), 'c_dense': np.diag(sv)})
