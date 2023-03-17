# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymor.tools.random import new_rng


class LoewnerReductor(BasicObject):
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
        the splitting rule. A user-defined partitioning can be defined by passing a tuple of the
        left and right indices. Defaults to `even-odd`.
    ordering
        The ordering with respect to which the splitting rule is executed. Can be either
        'magnitude', 'random' or 'regular'. Defaults to 'regular'.
    conjugate
        Whether to guarantee realness of reduced |LTIModel| by keeping complex conjugates in the
        same partitioning or not. If `True` will automatically generate conjugate data if necessary.
    MIMO_handling
        Option indicating how to treat MIMO systems. Can be:

        - `'random'` for using random tangential directions.
        - `'full'` for fully interpolating all input-output pairs.
        - Tuple `(ltd, rtd)` where `ltd` corresponds to left and `rtd` to right tangential
          directions.
    """

    def __init__(self, s, Hs, partitioning='even-odd', ordering='regular', conjugate=True, MIMO_handling='random'):
        self.__auto_init(locals())
        self.loewner_svds = None

    def reduce(self, r=None, tol=1e-7):
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
        L, Ls, V, W = loewner_quadruple(self.s, self.Hs, partitioning=self.partitioning, ordering=self.ordering,
                                        conjugate=self.conjugate, MIMO_handling=self.MIMO_handling)

        if self.loewner_svds is None:
            LhLs = np.hstack([L, Ls])
            Y, S1, _ = spla.svd(LhLs, full_matrices=False)
            LvLs = np.vstack([L, Ls])
            _, S2, Xh = spla.svd(LvLs, full_matrices=False)
            self.loewner_svds = (Y, S1, S2, Xh)
        else:
            Y, S1, S2, Xh = self.loewner_svds

        r1 = len(S1[S1 > tol])
        r2 = len(S2[S2 > tol])
        if r is None or r > r1 or r > r2:
            if r1 != r2:
                self.logger.warning(f'Mismatch in numerical rank of stacked Loewner matrices ({r1} and {r2}).'
                                    ' Consider increasing tol, specifying r or changing the partitioning.')
                r = (r1 + r2) // 2
            else:
                r = r1

        Yhr = Y[:, :r].conj().T
        Xr = Xh[:r, :].conj().T

        B = Yhr @ V
        C = W @ Xr
        E = - Yhr @ L @ Xr
        A = - Yhr @ Ls @ Xr

        if self.conjugate:
            A, B, C, E = A.real, B.real, C.real, E.real

        return LTIModel.from_matrices(A, B, C, D=None, E=E)


def _partition_frequencies(s, Hs, partitioning='even-odd', ordering='regular', conjugate=True):
    """Create a frequency partitioning."""
    # must keep complex conjugate frequencies in the same partioning
    if conjugate:
        # partition frequencies corresponding to positive imaginary part
        pimidx = np.where(np.imag(s) > 0)[0]

        # treat real-valued samples separately in order to ensure balanced splitting
        ridx = np.where(np.imag(s) == 0)[0]

        if ordering == 'magnitude':
            pimidx_sort = np.argsort([np.linalg.norm(Hs[i]) for i in pimidx])
            pimidx_ordered = pimidx[pimidx_sort]
            ridx_sort = np.argsort([np.linalg.norm(Hs[i]) for i in ridx])
            ridx_ordered = ridx[ridx_sort]
        elif ordering == 'random':
            rng = new_rng(0)
            rng.shuffle(pimidx)
            pimidx_ordered = pimidx
            rng.shuffle(ridx)
            ridx_ordered = ridx
        elif ordering == 'regular':
            pimidx_ordered = pimidx
            ridx_ordered = ridx

        if partitioning == 'even-odd':
            left = np.concatenate((ridx_ordered[::2], pimidx_ordered[::2]))
            right = np.concatenate((ridx_ordered[1::2], pimidx_ordered[1::2]))
        elif partitioning == 'half-half':
            pim_split = np.array_split(pimidx_ordered, 2)
            r_split = np.array_split(ridx_ordered, 2)
            left = np.concatenate((r_split[0], pim_split[0]))
            right = np.concatenate((r_split[1], pim_split[1]))

        l_cc = np.array([], dtype=int)
        for le in left:
            if np.imag(s[le]) != 0:
                l_cc = np.concatenate((l_cc, np.where(s == s[le].conj())[0]))
        left = np.concatenate((left, l_cc))

        r_cc = np.array([], dtype=int)
        for ri in right:
            if np.imag(s[ri]) != 0:
                r_cc = np.concatenate((r_cc, np.where(s == s[ri].conj())[0]))
        right = np.concatenate((right, r_cc))

        return (left, right)
    else:
        if ordering == 'magnitude':
            idx = np.argsort([np.linalg.norm(Hs[i]) for i in len(Hs[0])])
        elif ordering == 'random':
            rng = new_rng(0)
            idx = rng.permutation(s.shape[0])
        elif ordering == 'regular':
            idx = np.arange(s.shape[0])

        if partitioning == 'even-odd':
            return (idx[::2], idx[1::2])
        elif partitioning == 'half-half':
            idx_split = np.array_split(idx, 2)
            return (idx_split[0], idx_split[1])


def loewner_quadruple(s, Hs, partitioning='even-odd', ordering='regular', conjugate=True, MIMO_handling='random'):
    r"""Constructs a Loewner quadruple as |NumPy arrays|.

    The Loewner quadruple :cite:`ALI17`

    .. math::
        (\mathbb{L},\mathbb{L}_s,V,W)

    consists of the Loewner matrix :math:`\mathbb{L}`, the shifted Loewner matrix
    :math:`\mathbb{L}_s`, left interpolation data :math:`V` and right interpolation data :math:`W`.

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
        the splitting rule. A user-defined partitioning can be defined by passing a tuple of the
        left and right indices. Defaults to `even-odd`.
    ordering
        The ordering with respect to which the splitting rule is executed. Can be either
        'magnitude', 'random' or 'regular'. Defaults to 'regular'.
    conjugate
        Whether to guarantee realness of reduced |LTIModel| by keeping complex conjugates in the
        same partitioning or not. If `True` will automatically generate conjugate data if necessary.
    MIMO_handling
        Option indicating how to treat MIMO systems. Can be:

        - `'random'` for using random tangential directions.
        - `'full'` for fully interpolating all input-output pairs.
        - Tuple `(ltd, rtd)` where `ltd` corresponds to left and `rtd` to right tangential
          directions.

    Returns
    -------
    L
        Loewner matrix.
    Ls
        Shifted Loewner matrix.
    V
        Left interpolation data.
    W
        Right interpolation data.
    """
    assert isinstance(s, np.ndarray)
    if hasattr(Hs, 'transfer_function'):
        Hs = Hs.transfer_function
    assert isinstance(Hs, TransferFunction) or isinstance(Hs, np.ndarray) or isinstance(Hs, list)

    assert partitioning in ('even-odd', 'half-half') or len(partitioning) == 2
    assert ordering in ('magnitude', 'random', 'regular')

    if isinstance(Hs, TransferFunction):
        Hss = np.empty((len(s), Hs.dim_output, Hs.dim_input), dtype=s[0].dtype)
        for i, ss in enumerate(s):
            Hss[i] = Hs.eval_tf(ss)
        Hs = Hss
    else:
        assert Hs.shape[0] == len(s)

    # ensure that complex sampling values appear in complex conjugate pairs
    if conjugate:
        # if user provides paritioning sizes, make sure they are adjusted
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
            partitioning = (p0, p1)
        else:
            for i, ss in enumerate(s):
                if np.conj(ss) not in s:
                    s = np.append(s, np.conj(ss))
                    Hs = np.append(Hs, np.conj(Hs[i])[np.newaxis, ...], axis=0)

    ip, jp = _partition_frequencies(s, Hs, partitioning, ordering, conjugate) \
        if isinstance(partitioning, str) else partitioning

    if len(Hs.shape) > 1:
        dim_output = Hs.shape[1]
        dim_input = Hs.shape[2]
    else:
        dim_input = 1
        dim_output = 1

    if dim_input == dim_output == 1:
        Hs = np.squeeze(Hs)
        L = Hs[ip][:, np.newaxis] - Hs[jp][np.newaxis]
        L /= s[ip][:, np.newaxis] - s[jp][np.newaxis]
        Ls = (s[ip] * Hs[ip])[:, np.newaxis] - (s[jp] * Hs[jp])[np.newaxis]
        Ls /= s[ip][:, np.newaxis] - s[jp][np.newaxis]
        V = Hs[ip][:, np.newaxis]
        W = Hs[jp][np.newaxis]
    else:
        if MIMO_handling == 'full':
            L = Hs[ip][:, np.newaxis] - Hs[jp][np.newaxis]
            L /= (s[ip][:, np.newaxis] - s[jp][np.newaxis])[:, :, np.newaxis, np.newaxis]
            Ls = (s[ip, np.newaxis, np.newaxis] * Hs[ip])[:, np.newaxis] \
                - (s[jp, np.newaxis, np.newaxis] * Hs[jp])[np.newaxis]
            Ls /= (s[ip][:, np.newaxis] - s[jp][np.newaxis])[:, :, np.newaxis, np.newaxis]
            V = Hs[ip][:, np.newaxis]
            W = Hs[jp][np.newaxis]
        else:
            if MIMO_handling == 'random':
                rng = new_rng(0)
                ltd = rng.normal(size=(len(ip), dim_output))
                rtd = rng.normal(size=(dim_input, len(jp)))
            elif len(MIMO_handling) == 2:
                ltd = MIMO_handling[0]
                rtd = MIMO_handling[1]
                assert ltd.shape == (len(ip), dim_output)
                assert rtd.shape == (dim_input, len(jp))
            L = np.empty((len(ip), len(jp)), dtype=np.complex_)
            Ls = np.empty((len(ip), len(jp)), dtype=np.complex_)
            V = np.empty((len(ip), dim_input), dtype=np.complex_)
            W = np.empty((dim_output, len(jp)), dtype=np.complex_)
            for i, si in enumerate(ip):
                for j, sj in enumerate(jp):
                    L[i, j] = ltd[i] @ (Hs[si] - Hs[sj]) @ rtd[:, j] / (s[si] - s[sj])
                    Ls[i, j] = ltd[i] @ (s[si] * Hs[si] - s[sj] * Hs[sj]) @ rtd[:, j] / (s[si] - s[sj])
                V[i, :] = Hs[si].T @ ltd[i]
            for j, sj in enumerate(jp):
                W[:, j] = Hs[sj] @ rtd[:, j]

    # transform the system to have real matrices
    if conjugate:
        TL = np.zeros((len(ip), len(ip)), dtype=np.complex_)
        for i, si in enumerate(ip):
            if s[si].imag == 0:
                TL[i, i] = 1
            else:
                j = np.argmin(np.abs(s[ip] - s[si].conjugate()))
                if i < j:
                    TL[i, i] = 1
                    TL[i, j] = 1
                    TL[j, i] = -1j
                    TL[j, j] = 1j

        TR = np.zeros((len(jp), len(jp)), dtype=np.complex_)
        for i, si in enumerate(jp):
            if s[si].imag == 0:
                TR[i, i] = 1
            else:
                j = np.argmin(np.abs(s[jp] - s[si].conjugate()))
                if i < j:
                    TR[i, i] = 1
                    TR[i, j] = 1
                    TR[j, i] = -1j
                    TR[j, j] = 1j
        TR = TR / np.sqrt(2)
        TL = TL / np.sqrt(2)

        if MIMO_handling == 'full':
            L = np.tensordot(TL, L, axes=(1, 0))
            L = np.tensordot(L, TR.conj().T, axes=(1, 0))
            L = L.real

            Ls = np.tensordot(TL, Ls, axes=(1, 0))
            Ls = np.tensordot(Ls, TR.conj().T, axes=(1, 0))
            Ls = Ls.real

            V = np.tensordot(TL, V, axes=(1, 0)).real
            W = np.tensordot(W, TR.conj().T, axes=(1, 0)).real
        else:
            L = (TL @ L @ TR.conj().T).real
            Ls = (TL @ Ls @ TR.conj().T).real
            V = (TL @ V).real
            W = (W @ TR.conj().T).real

    if MIMO_handling == 'full':
        L = np.reshape(L, (len(ip)*dim_output, len(jp)*dim_input))
        Ls = np.reshape(Ls, (len(ip)*dim_output, len(jp)*dim_input))
        V = np.reshape(V, (len(ip)*dim_output, dim_input))
        W = np.reshape(W, (dim_output, len(jp)*dim_input))

    return L, Ls, V, W