# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
import numpy as np
import scipy.linalg as spla

from pymor.algorithms.projection import project
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.cache import cached, CacheableObject
from pymor.models.iosys import LTIModel
from pymor.operators.numpy import NumpyHankelOperator, NumpyMatrixOperator


class ERAReductor(CacheableObject):
    r"""Eigensystem Realization Algorithm reductor.

    Constructs a (reduced) realization that matches the Markov parameters optimally for given order
    by orthogonal factorization of the Hankel matrix of Markov parameters (see :cite:`K78`).

    For a large number if inputs and/or outputs, the factorization of the Hankel matrix can be
    accelerated by tangential projections of the Markov parameters first (see :cite:`KG16`).

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
        Kung's stability assumption :cite:`K78`. Defaults to `True`.
    """

    cache_region = 'memory'

    def __init__(self, data, sampling_time, force_stability=True):
        assert sampling_time > 0
        if data.ndim == 1:
            data = data.reshape(-1, 1, 1)
        assert data.ndim == 3
        if force_stability:
            data = np.concatenate([data, np.zeros_like(data)[1:]], axis=0)
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
        return s2, W2.conj().T

    def _project_markov_parameters(self, l1, l2):
        mats = [self.output_projector(l1).conj().T, self.data] if l1 else [self.data]
        mats = [*mats, self.input_projector(l2)] if l2 else mats
        s1 = ('lp,', 'l') if l1 else ('', 'p')
        s2 = (',mk', 'k') if l2 else ('', 'm')
        self.logger.info('Projecting Markov parameters ...')
        einstr = s1[0] + 'npm' + s2[0] + '->n' + s1[1] + s2[1]
        return np.einsum(einstr, *mats, optimize='optimal')

    @cached
    def _sv_U_V(self, l1, l2):
        h = self._project_markov_parameters(l1, l2) if l1 or l2 else self.data
        self.logger.info(f'Computing SVD of the {"projected " if l1 or l2 else ""}Hankel matrix ...')
        U, sv, V = spla.svd(to_matrix(NumpyHankelOperator(h)), full_matrices=False)
        return sv, U.T, V

    def output_projector(self, l1):
        assert isinstance(l1, int) and l1 <= self.data.shape[1]
        self.logger.info(f'Constructing output projector ({l1} tangential directions) ...')
        return self._s1_W1()[1][:, :l1]

    def input_projector(self, l2):
        assert isinstance(l2, int) and l2 <= self.data.shape[2]
        self.logger.info(f'Constructing input projector ({l2} tangential directions) ...')
        return self._s2_W2()[1][:, :l2]

    def error_bounds(self, l1=None, l2=None):
        sv = self._sv_U_V(l1, l2)[0]
        _, p, m = self.data.shape
        err = (np.sqrt(np.arange(len(sv)) + (l2 or m) + (l1 or p)) * sv)[1:]
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
        s, p, m = self.data.shape
        assert l1 is None or isinstance(l1, int) and l1 <= p
        assert l2 is None or isinstance(l2, int) and l2 <= m
        assert r is None or 0 < r <= min((l1 or p), (l2 or m)) * (s // 2 + 1)

        sv, U, V = self._sv_U_V(l1, l2)

        if tol is not None:
            error_bounds = self.error_bounds(l1=l1, l2=l2)
            r_tol = np.argmax(error_bounds <= tol) + 1
            r = r_tol if r is None else min(r, r_tol)

        sv, U, V = sv[:r], U[:r], V[:r]

        self.logger.info(f'Constructing reduced realization of order {r} ...')
        sqS = np.diag(np.sqrt(sv))
        Zo = U.T @ sqS
        A = NumpyMatrixOperator(spla.pinv(Zo[: -(l1 or p)]) @ Zo[(l1 or p):])
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

        return LTIModel(A, B, C, sampling_time=self.sampling_time,
                        presets={'o_dense': np.diag(sv), 'c_dense': np.diag(sv)})
