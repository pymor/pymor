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
        |Numpy Array| of shape (n, p, m) or |TransferFunction| resembling the transfer function samples.
    partitioning
        `str` or `tuple` of length 2. Strings can either be 'even-odd' or 'half-half' defining
        the splitting rule. A user-defined partitioning can be defined by passing a tuple of the left and
        right indices. Defaults to `even-odd`.
    ordering
        The ordering with respect to which the splitting rule is executed. Can be either 'magnitude',
        'random' or 'regular'. Defaults to 'regular'.
    ltd
        |Numpy Array| representing left tangential directions. If `None` tangential directions will be
        chosen randomly (normal distribution).
    rtd
        |Numpy Array| representing right tangential directions. If `None` tangential directions will be
        chosen randomly (normal distribution).
    """

    def __init__(self, s, Hs, partitioning='even-odd', ordering='regular', ltd=None, rtd=None):
        self.__auto_init(locals())

    def reduce(self, tol=1e-7):
        """Reduce using Loewner framework.

        Parameters
        ----------
        tol
            Truncation tolerance for rank of Loewner matrices.

        Returns
        -------
        rom
            Reduced |LTIModel|.
        """
        L, Ls, V, W = loewner_quadruple(self.s, self.Hs, partitioning='even-odd', ordering='regular', ltd=None, rtd=None)
        LLS = np.vstack([L, Ls])
        Y, S, _ = spla.svd(LLS, full_matrices=False)
        _, _, Xh = spla.svd(LLS.T, full_matrices=False)

        r = len(S[S > tol])
        Yr = Y[:r, :]
        Xhr = Xh[:, :r]
        B = Yr @ V
        C = W @ Xhr.conj()
        E = - Yr @ L @ Xhr.conj()
        A = - Yr @ Ls @ Xhr.conj()

        return LTIModel.from_matrices(A, B, C, D=None, E=E)


def _partition_frequencies(s, Hs, partitioning='even-odd', ordering='regular'):
    """Create a frequency partitioning."""
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
        np.random.shuffle(pimidx)
        pimidx_ordered = pimidx
        np.random.permutation(ridx)
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


def loewner_quadruple(s, Hs, partitioning='even-odd', ordering='regular', ltd=None, rtd=None):
    r"""Constructs a Loewner quadruple as |NumPy arrays|.

    The Loewner quadruple :cite:`ALI17`

    .. math::
        (\mathbb{L},\mathbb{L}_s,V,W)

    consists of the Loewner matrix :math:`\mathbb{L}`, the shifted Loewner matrix :math:`\mathbb{L}_s`,
    left interpolation data :math:`V` and right interpolation data :math:`W`.

    Parameters
    ----------
    s
        |Numpy Array| of shape (n,) containing the frequencies.
    Hs
        |Numpy Array| of shape (n, p, m) or |TransferFunction| resembling the transfer function samples.
    partitioning
        `str` or `tuple` of length 2. Strings can either be 'even-odd' or 'half-half' defining
        the splitting rule. A user-defined partitioning can be defined by passing a tuple of the left and
        right indices. Defaults to `even-odd`.
    ordering
        The ordering with respect to which the splitting rule is executed. Can be either 'magnitude',
        'random' or 'regular'. Defaults to 'regular'.
    ltd
        |Numpy Array| representing left tangential directions. If `None` tangential directions will be
        chosen randomly (normal distribution).
    rtd
        |Numpy Array| representing right tangential directions. If `None` tangential directions will be
        chosen randomly (normal distribution).

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
    assert isinstance(Hs, TransferFunction) or isinstance(Hs, np.ndarray) and Hs.shape[0] == s.shape[0]

    assert partitioning in ('even-odd', 'half-half') or len(partitioning) == 2
    assert ordering in ('magnitude', 'random', 'regular')

    if isinstance(Hs, TransferFunction):
        Hss = np.empty((len(s), Hs.dim_output, Hs.dim_input), dtype=s[0].dtype)
        for i, ss in enumerate(s):
            Hss[i] = Hs.eval_tf(ss)
        Hs = Hss
    else:
        assert len(Hs[0]) == len(s)

    # ensure that complex sampling values appear in complex conjugate pairs
    for i, ss in enumerate(s):
        if np.conj(ss) not in s:
            s = np.append(s, np.conj(ss))
            Hs = np.append(Hs, np.conj(Hs[i])[None, ...], axis=0)

    ip, jp = _partition_frequencies(s, Hs, partitioning, ordering) if isinstance(partitioning, str) else partitioning

    if len(Hs.shape) > 1:
        dim_output = Hs.shape[1]
        dim_input = Hs.shape[2]
        if ltd is None:
            rng = new_rng(0)
            ltd = rng.normal(size=(len(ip), dim_output))
        else:
            assert ltd.shape == (len(ip), dim_output)

        if rtd is None:
            rng = new_rng(0)
            rtd = rng.normal(size=(dim_input, len(jp)))
        else:
            assert rtd.shape == (dim_input, len(jp))
    else:
        dim_input = 1
        dim_output = 1

    if rtd is None and ltd is None:
        L = Hs[ip][:, np.newaxis] - Hs[jp][np.newaxis]
        L /= s[ip][:, np.newaxis] - s[jp][np.newaxis]
        Ls = (s[ip] * Hs[ip])[:, np.newaxis] - (s[jp] * Hs[jp])[np.newaxis]
        Ls /= s[ip][:, np.newaxis] - s[jp][np.newaxis]
        V = Hs[ip][:, np.newaxis]
        W = Hs[jp][np.newaxis]
    else:
        L = np.empty((len(ip), len(jp)), dtype=np.complex_)
        Ls = np.empty((len(ip), len(jp)), dtype=np.complex_)
        V = np.empty((len(ip), dim_input), dtype=np.complex_)
        W = np.empty((dim_output, len(jp)), dtype=np.complex_)
        for i, si in enumerate(ip):
            for j, sj in enumerate(jp):
                L[i, j] = ltd[i] @ (Hs[si] - Hs[sj]) @ rtd[:,j] / (s[si] - s[sj])
                Ls[i, j] = ltd[i] @ (s[si] * Hs[si] - s[sj] * Hs[sj]) @ rtd[:,j] / (s[si] - s[sj])
            V[i, :] = Hs[si].T @ ltd[i]
        for j, sj in enumerate(jp):
            W[:, j] = Hs[sj] @ rtd[:,j]

    # transform the system to have real matrices
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

    L = (TL @ L @ TR.conj().T).real
    Ls = (TL @ Ls @ TR.conj().T).real
    V = (TL @ V).real
    W = (W @ TR.conj().T).real

    return L, Ls, V, W
