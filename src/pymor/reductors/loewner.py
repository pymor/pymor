#!/usr/bin/env python3
import numpy as np

from pymor.models.transfer_function import TransferFunction


def _partition_frequencies(s, Hs, partitioning='even-odd', ordering='regular'):
    """Create a frequency partitioning."""
    if ordering == 'magnitude':
        idx = np.argsort(np.linalg.norm(Hs), axis=(1, 2))
    elif ordering == 'random':
        idx = np.random.permutation(s.shape[0])
    elif ordering == 'regular':
        idx = np.arange(s.shape[0])

    if partitioning == 'even-odd':
        return idx[::2], idx[1::2]
    elif partitioning == 'half-half':
        return np.array_split(idx, 2)
    elif partitioning == 'same':
        return idx, idx


def _construct_loewner_matrix(s, Hs, dHs=None, shifted=False, partitioning='even-odd', ordering='regular'):
    """Constructs a (shifted) Loewner matrix as a |NumPy array|."""
    assert isinstance(s, np.ndarray)
    if hasattr(Hs, 'transfer_function'):
        Hs = Hs.transfer_function
    assert isinstance(Hs, TransferFunction) or isinstance(Hs, np.ndarray) and Hs.shape[0] == s.shape[0]
    assert dHs is None or isinstance(dHs, np.ndarray)

    assert isinstance(shifted, bool)
    assert partitioning in ('even-odd', 'half-half', 'same') or len(partitioning) == 2
    if partitioning == 'same':
        raise NotImplementedError
    assert ordering in ('magnitude', 'random', 'regular')

    # compute derivatives if needed
    if not isinstance(partitioning, str) and np.any(np.isin(*partitioning)) or partitioning == 'same':
        if isinstance(Hs, TransferFunction):
            assert Hs.dtf is not None
            dHs = np.stack([Hs.eval_dtf(si) for si in s])
        else:
            assert isinstance(dHs, np.ndarray) and Hs.shape == dHs.shape

    if isinstance(Hs, TransferFunction):
        Hs = np.stack([Hs.eval_tf(si) for si in s])

    i, j = _partition_frequencies(s, Hs, partitioning, ordering) if isinstance(partitioning, str) else partitioning

    if shifted:
        L = (np.expand_dims(s[i], axis=(1, 2)) * Hs[i])[:, np.newaxis] - (np.expand_dims(s[j], axis=(1, 2)) * Hs[j])[np.newaxis]
    else:
        L = Hs[i][:, np.newaxis] - Hs[j][np.newaxis]

    L /= np.expand_dims(np.subtract.outer(s[i], s[j]), axis=(2, 3))

    return np.concatenate(np.concatenate(L, axis=1), axis=1)
