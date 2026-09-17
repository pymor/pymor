# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.loewner import (
    sample_transfer_function,
    complete_conjugate_pairs,
    loewner_quadruple,
    partition_frequencies,
)
from pymor.core.cache import CacheableObject, cached
from pymor.models.iosys import LTIModel
from pymor.tools.random import new_rng


class LoewnerReductor(CacheableObject):
    """Reductor based on Loewner interpolation framework.

    The reductor implements interpolation based on the Loewner framework as in :cite:`ALI17`.

    Parameters
    ----------
    s
        |NumPy array| of shape `(n,)` containing the frequencies.
    Hs
        |NumPy array| of transfer function samples, of shape `(n, p, m)` for MIMO systems
        with `p` outputs and `m` inputs or `(n,)` for SISO systems.
        Use :meth:`generate_samples` to generate data beforehand.
    partitioning
        `str` or `tuple` of length 2. Strings can either be `'even-odd'` or `'half-half'` defining
        the partitioning rule. A user-defined partitioning can be defined by passing a tuple of the
        left and right indices. Defaults to `'even-odd'`.
    ordering
        The ordering with respect to which the partitioning rule is executed. Can be either
        `'magnitude'`, `'random'` or `'regular'`. Defaults to `'regular'`.
    force_real
        Whether to guarantee realness of reduced |LTIModel| by keeping complex conjugates in the
        same partitioning or not. If `True` will automatically generate conjugate data if necessary.
    mimo_handling
        Option indicating how to treat MIMO systems. Can be:

        - `'random'` for using random tangential directions.
        - `'full'` for fully interpolating all input-output pairs.
        - Tuple `(ltd, rtd)` where `ltd` corresponds to left and `rtd` to right tangential
          directions. If `force_real=True`, directions at conjugate nodes must be conjugates.
    """

    cache_region = 'memory'

    generate_samples = staticmethod(sample_transfer_function)

    def __init__(self, s, Hs, partitioning='even-odd', ordering='regular', force_real=True, mimo_handling='full'):
        assert isinstance(s, np.ndarray)
        if isinstance(partitioning, str):
            assert partitioning in ('even-odd', 'half-half'), f'Unknown partitioning: {partitioning}.'
        else:
            assert len(partitioning) == 2, 'partitioning must contain left and right indices.'
            assert len(partitioning[0]) + len(partitioning[1]) == len(s), 'partitioning must cover all samples.'
        assert ordering in ('magnitude', 'random', 'regular'), f'Unknown ordering: {ordering}.'

        Hs = np.asarray(Hs)
        assert Hs.ndim in (1, 3), 'Hs must contain sample data; use generate_samples to sample a model.'
        assert Hs.shape[0] == len(s)

        if force_real:
            num_samples = len(s)
            s, Hs, source_indices = complete_conjugate_pairs(s, Hs, np.arange(num_samples))
            # if user-defined partitioning is supplied, update this partitioning with added samples
            if isinstance(partitioning, tuple):
                added_sources = source_indices[num_samples:]
                partitioning = tuple(
                    np.concatenate((indices, num_samples + np.flatnonzero(np.isin(added_sources, indices))))
                    for indices in partitioning
                )
            if len(s) != num_samples:
                self.logger.info(f'Added {len(s) - num_samples} complex conjugates to the data.')

        if Hs.ndim == 3:
            self.dim_output = Hs.shape[1]
            self.dim_input = Hs.shape[2]
            if self.dim_output == self.dim_input == 1:
                Hs = Hs[:, 0, 0]
        else:
            self.dim_output = 1
            self.dim_input = 1

        self.s = s
        self.Hs = Hs
        self.partitioning = partitioning
        self.ordering = ordering
        self.conjugate = conjugate
        self.mimo_handling = mimo_handling

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

        return LTIModel.from_matrices(A, B, C, E=E)

    @cached
    def loewner_quadruple(self):
        r"""Construct a Loewner quadruple as |NumPy arrays|.

        The Loewner quadruple :cite:`ALI17`

        .. math::
            (\mathbb{L},\mathbb{L}_s,V,W)

        consists of the Loewner matrix :math:`\mathbb{L}`, the shifted Loewner matrix
        :math:`\mathbb{L}_s`, left interpolation data :math:`V` and right interpolation
        data :math:`W`.
        """
        ip, jp = partition_frequencies(self.s, self.Hs, partitioning=self.partitioning,
                                       ordering=self.ordering, force_real=self.force_real)
        left_directions = right_directions = None
        if self.dim_input != 1 or self.dim_output != 1:
            if self.mimo_handling == 'random':
                rng = new_rng(seed_seq=0)
                # Use the same directions at all nodes to preserve conjugate symmetry.
                left_directions = np.tile(rng.normal(size=(1, self.dim_output)), (len(ip), 1))
                right_directions = np.tile(rng.normal(size=(1, self.dim_input)), (len(jp), 1))
            elif self.mimo_handling != 'full':
                left_directions, right_directions = self.mimo_handling
                assert left_directions.shape == (len(ip), self.dim_output)
                assert right_directions.shape == (self.dim_input, len(jp))
                right_directions = right_directions.T

        return loewner_quadruple(
            self.s[ip], self.s[jp], self.Hs[ip], self.Hs[jp],
            left_directions=left_directions, right_directions=right_directions, force_real=self.force_real,
        )

    @cached
    def _loewner_svds(self, L, Ls):
        """Compute SVDs of stacked Loewner matrices."""
        LhLs = np.hstack([L, Ls])
        Y, S1, _ = spla.svd(LhLs, full_matrices=False)
        LvLs = np.vstack([L, Ls])
        _, S2, Xh = spla.svd(LvLs, full_matrices=False)

        return Y, S1, S2, Xh
