# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.core.cache import CacheableObject, cached
from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymor.tools.random import new_rng


class LoewnerReductor(CacheableObject):
    """Reductor based on Loewner interpolation framework.

    The reductor implements interpolation based on the Loewner framework as in :cite:`ALI17`.

    Parameters
    ----------
    s
        |Numpy Array| of shape (n,) containing the frequencies.
    Hs
        |Numpy Array| of shape (n, p, m) for MIMO systems with p outputs and m inputs or
        |Numpy Array| of shape (n,) for SISO systems where the |Numpy Arrays| resemble the transfer
        function samples. Alternatively, |TransferFunction| or `Model` with `transfer_function`
        attribute.
    partitioning
        `str` or `tuple` of length 2. Strings can either be 'even-odd' or 'half-half' defining
        the partitioning rule. A user-defined partitioning can be defined by passing a tuple of the
        left and right indices. Defaults to `even-odd`.
    ordering
        The ordering with respect to which the partitioning rule is executed. Can be either
        'magnitude', 'random' or 'regular'. Defaults to 'regular'.
    conjugate
        Whether to guarantee realness of reduced |LTIModel| by keeping complex conjugates in the
        same partitioning or not. If `True` will automatically generate conjugate data if necessary.
    mimo_handling
        Option indicating how to treat MIMO systems. Can be:

        - `'random'` for using random tangential directions.
        - `'full'` for fully interpolating all input-output pairs.
        - Tuple `(ltd, rtd)` where `ltd` corresponds to left and `rtd` to right tangential
          directions.
    """

    cache_region = 'memory'

    def __init__(self, s, Hs, partitioning='even-odd', ordering='regular', conjugate=True, mimo_handling='random'):
        assert isinstance(s, np.ndarray)
        if hasattr(Hs, 'transfer_function'):
            Hs = Hs.transfer_function
        assert isinstance(Hs, (TransferFunction, np.ndarray, list))

        assert partitioning in ('even-odd', 'half-half') \
            or len(partitioning) == 2 \
            and len(partitioning[0]) + len(partitioning[1]) == len(s)
        assert ordering in ('magnitude', 'random', 'regular')

        if isinstance(Hs, TransferFunction):
            Hss = np.empty((len(s), Hs.dim_output, Hs.dim_input), dtype=s[0].dtype)
            for i, ss in enumerate(s):
                Hss[i] = Hs.eval_tf(ss)
            Hs = Hss
        else:
            Hs = Hs
            assert Hs.shape[0] == len(s)

        # ensure that complex sampling values appear in complex conjugate pairs
        if conjugate:
            # if user provides partitioning sizes, make sure they are adjusted
            if isinstance(partitioning, tuple):
                p0 = partitioning[0]
                p1 = partitioning[1]
                for i, ss in enumerate(s):
                    if np.conj(ss) not in s:
                        s = np.append(s, np.conj(ss))
                        Hs = np.append(Hs, np.conj(Hs[i])[np.newaxis, ...], axis=0)
                        if i in p0:
                            p0 = np.append(p0, len(s)-1)
                        else:
                            p1 = np.append(p1, len(s)-1)
                if len(p0) != len(partitioning[0]) or len(p1) != len(partitioning[1]):
                    self.logger.info('Added complex conjugates to partitionings. '
                                     f'New partitioning sizes are ({len(p0)}, {len(p1)}).')
                partitioning = (p0, p1)
            else:
                s_new = s
                for i, ss in enumerate(s):
                    if np.conj(ss) not in s:
                        s_new = np.append(s_new, np.conj(ss))
                        Hs = np.append(Hs, np.conj(Hs[i])[np.newaxis, ...], axis=0)
                if len(s) != len(s_new):
                    self.logger.info(f'Added {len(s_new) - len(s)} complex conjugates to the data.')
                s = s_new

        if len(Hs.shape) > 1:
            self.dim_output = Hs.shape[1]
            self.dim_input = Hs.shape[2]
            if self.dim_output == self.dim_output == 1:
                Hs = np.squeeze(Hs)
        else:
            self.dim_output = 1
            self.dim_input = 1

        self.__auto_init(locals())

    def reduce(self, r=None, tol=1e-12):
        """Reduce using Loewner framework.

        Parameters
        ----------
        r
            Integer for target order of reduced model. If an interpolant with order less than r
            exists then the output will have the minimal order of an interpolant. Otherwise, the
            output will be an |LTIModel| with order r. If `None` the order of the reduced model will
            be the minimal order of an interpolant.
        tol
            Truncation tolerance for rank of Loewner matrices.

        Returns
        -------
        rom
            Reduced |LTIModel|.
        """
        L, Ls, V, W = self.loewner_quadruple()
        Y, S1, S2, Xh = self._loewner_svds(L, Ls)

        r1 = len(S1[S1/S1[0] > tol])
        r2 = len(S2[S2/S2[0] > tol])
        if r is None or r > r1 or r > r2:
            if r1 != r2:
                self.logger.warning(f'Mismatch in numerical rank of stacked Loewner matrices ({r1} and {r2}).'
                                    ' Consider increasing tol, specifying r or changing the partitioning.')
                r = min(r1, r2)
            else:
                r = r1

        Yhr = Y[:, :r].conj().T
        Xr = Xh[:r, :].conj().T

        B = Yhr @ V
        C = W @ Xr
        E = -Yhr @ L @ Xr
        A = -Yhr @ Ls @ Xr

        if self.conjugate:
            A, B, C, E = A.real, B.real, C.real, E.real

        return LTIModel.from_matrices(A, B, C, D=None, E=E)


    def _partition_frequencies(self):
        """Create a frequency partitioning."""
        # must keep complex conjugate frequencies in the same partitioning
        if self.conjugate:
            # partition frequencies corresponding to positive imaginary part
            pimidx = np.where(self.s.imag > 0)[0]

            # treat real-valued samples separately in order to ensure balanced partitioning
            ridx = np.where(self.s.imag == 0)[0]

            if self.ordering == 'magnitude':
                pimidx_sort = np.argsort([np.linalg.norm(self.Hs[i]) for i in pimidx])
                pimidx_ordered = pimidx[pimidx_sort]
                ridx_sort = np.argsort([np.linalg.norm(self.Hs[i]) for i in ridx])
                ridx_ordered = ridx[ridx_sort]
            elif self.ordering == 'random':
                rng = new_rng(0)
                rng.shuffle(pimidx)
                pimidx_ordered = pimidx
                rng.shuffle(ridx)
                ridx_ordered = ridx
            elif self.ordering == 'regular':
                pimidx_ordered = pimidx
                ridx_ordered = ridx

            if self.partitioning == 'even-odd':
                left = np.concatenate((ridx_ordered[::2], pimidx_ordered[::2]))
                right = np.concatenate((ridx_ordered[1::2], pimidx_ordered[1::2]))
            elif self.partitioning == 'half-half':
                pim_split = np.array_split(pimidx_ordered, 2)
                r_split = np.array_split(ridx_ordered, 2)
                left = np.concatenate((r_split[0], pim_split[0]))
                right = np.concatenate((r_split[1], pim_split[1]))

            l_cc = np.array([], dtype=int)
            for le in left:
                if self.s[le].imag != 0:
                    l_cc = np.concatenate((l_cc, np.where(self.s == self.s[le].conj())[0]))
            left = np.concatenate((left, l_cc))

            r_cc = np.array([], dtype=int)
            for ri in right:
                if self.s[ri].imag != 0:
                    r_cc = np.concatenate((r_cc, np.where(self.s == self.s[ri].conj())[0]))
            right = np.concatenate((right, r_cc))

            return (left, right)
        else:
            if self.ordering == 'magnitude':
                idx = np.argsort([np.linalg.norm(self.Hs[i]) for i in len(self.Hs[0])])
            elif self.ordering == 'random':
                rng = new_rng(0)
                idx = rng.permutation(self.s.shape[0])
            elif self.ordering == 'regular':
                idx = np.arange(self.s.shape[0])

            if self.partitioning == 'even-odd':
                return (idx[::2], idx[1::2])
            elif self.partitioning == 'half-half':
                idx_split = np.array_split(idx, 2)
                return (idx_split[0], idx_split[1])

    @cached
    def loewner_quadruple(self):
        r"""Constructs a Loewner quadruple as |NumPy arrays|.

        The Loewner quadruple :cite:`ALI17`

        .. math::
            (\mathbb{L},\mathbb{L}_s,V,W)

        consists of the Loewner matrix :math:`\mathbb{L}`, the shifted Loewner matrix
        :math:`\mathbb{L}_s`, left interpolation data :math:`V` and right interpolation
        data :math:`W`.

        Returns
        -------
        L
            Loewner matrix as a |NumPy array|.
        Ls
            Shifted Loewner matrix as a |NumPy array|.
        V
            Left interpolation data as a |NumPy array|.
        W
            Right interpolation data as a |NumPy array|.
        """
        ip, jp = self._partition_frequencies() if isinstance(self.partitioning, str) else self.partitioning

        if self.dim_input == self.dim_output == 1:
            L = self.Hs[ip][:, np.newaxis] - self.Hs[jp]
            L /= self.s[ip][:, np.newaxis] - self.s[jp]
            Ls = (self.s[ip] * self.Hs[ip])[:, np.newaxis] - self.s[jp] * self.Hs[jp]
            Ls /= self.s[ip][:, np.newaxis] - self.s[jp]
            V = self.Hs[ip][:, np.newaxis]
            W = self.Hs[jp][np.newaxis]
        else:
            if self.mimo_handling == 'full':
                L = self.Hs[ip][:, np.newaxis] - self.Hs[jp][np.newaxis]
                L /= (self.s[ip][:, np.newaxis] - self.s[jp][np.newaxis])[:, :, np.newaxis, np.newaxis]
                Ls = (self.s[ip, np.newaxis, np.newaxis] * self.Hs[ip])[:, np.newaxis] \
                    - (self.s[jp, np.newaxis, np.newaxis] * self.Hs[jp])[np.newaxis]
                Ls /= (self.s[ip][:, np.newaxis] - self.s[jp][np.newaxis])[:, :, np.newaxis, np.newaxis]
                V = self.Hs[ip][:, np.newaxis]
                W = self.Hs[jp][np.newaxis]
            else:
                if self.mimo_handling == 'random':
                    rng = new_rng(0)
                    ltd = rng.normal(size=(len(ip), self.dim_output))
                    rtd = rng.normal(size=(self.dim_input, len(jp)))
                elif len(self.mimo_handling) == 2:
                    ltd = self.mimo_handling[0]
                    rtd = self.mimo_handling[1]
                    assert ltd.shape == (len(ip), self.dim_output)
                    assert rtd.shape == (self.dim_input, len(jp))
                L = np.empty((len(ip), len(jp)), dtype=np.complex_)
                Ls = np.empty((len(ip), len(jp)), dtype=np.complex_)
                V = np.empty((len(ip), self.dim_input), dtype=np.complex_)
                W = np.empty((self.dim_output, len(jp)), dtype=np.complex_)
                for i, si in enumerate(ip):
                    for j, sj in enumerate(jp):
                        L[i, j] = ltd[i] @ (self.Hs[si] - self.Hs[sj]) @ rtd[:, j] / (self.s[si] - self.s[sj])
                        Ls[i, j] = ltd[i] @ (self.s[si] * self.Hs[si] - self.s[sj] * self.Hs[sj]) @ rtd[:, j] \
                            / (self.s[si] - self.s[sj])
                    V[i, :] = self.Hs[si].T @ ltd[i]
                for j, sj in enumerate(jp):
                    W[:, j] = self.Hs[sj] @ rtd[:, j]

        # transform the system to have real matrices
        if self.conjugate:
            TL = np.zeros((len(ip), len(ip)), dtype=np.complex_)
            for i, si in enumerate(ip):
                if self.s[si].imag == 0:
                    TL[i, i] = 1
                else:
                    j = np.argmin(np.abs(self.s[ip] - self.s[si].conjugate()))
                    if i < j:
                        TL[i, i] = 1
                        TL[i, j] = 1
                        TL[j, i] = -1j
                        TL[j, j] = 1j

            TR = np.zeros((len(jp), len(jp)), dtype=np.complex_)
            for i, si in enumerate(jp):
                if self.s[si].imag == 0:
                    TR[i, i] = 1
                else:
                    j = np.argmin(np.abs(self.s[jp] - self.s[si].conjugate()))
                    if i < j:
                        TR[i, i] = 1
                        TR[i, j] = 1
                        TR[j, i] = -1j
                        TR[j, j] = 1j
            TR = TR / np.sqrt(2)
            TL = TL / np.sqrt(2)

            if self.mimo_handling == 'full' and not self.dim_input == self.dim_output == 1:
                L = np.tensordot(TL, L, axes=(1, 0))
                L = np.tensordot(L, TR.conj().T, axes=(1, 0))
                L = L.real
                L = np.transpose(L, (0, 3, 1, 2))

                Ls = np.tensordot(TL, Ls, axes=(1, 0))
                Ls = np.tensordot(Ls, TR.conj().T, axes=(1, 0))
                Ls = Ls.real
                Ls = np.transpose(Ls, (0, 3, 1, 2))

                V = np.tensordot(TL, V, axes=(1, 0)).real
                W = np.tensordot(W, TR.conj().T, axes=(1, 0)).real
                W = np.transpose(W, (0, 3, 1, 2))
            else:
                L = (TL @ L @ TR.conj().T).real
                Ls = (TL @ Ls @ TR.conj().T).real
                V = (TL @ V).real
                W = (W @ TR.conj().T).real

        if self.mimo_handling == 'full' and not self.dim_input == self.dim_output == 1:
            L = np.concatenate(np.concatenate(L, axis=1), axis=1)
            Ls = np.concatenate(np.concatenate(Ls, axis=1), axis=1)
            V = np.concatenate(np.concatenate(V, axis=1), axis=1)
            W = np.concatenate(np.concatenate(W, axis=0), axis=1)

        return L, Ls, V, W

    @cached
    def _loewner_svds(self, L, Ls):
        """Compute SVDs of stacked Loewner matrices."""
        LhLs = np.hstack([L, Ls])
        Y, S1, _ = spla.svd(LhLs, full_matrices=False)
        LvLs = np.vstack([L, Ls])
        _, S2, Xh = spla.svd(LvLs, full_matrices=False)

        return Y, S1, S2, Xh
