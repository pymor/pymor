# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.loewner import (
    _sample_transfer_function,
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
        |Numpy Array| of shape (n,) containing the frequencies.
    Hs
        |Numpy Array| of shape (n, p, m) for MIMO systems with p outputs and m inputs or
        |Numpy Array| of shape (n,) for SISO systems where the |Numpy Arrays| resemble the transfer
        function samples. Use :meth:`generate_samples` to generate data beforehand.
    partitioning
        `str` or `tuple` of length 2. Strings can either be 'even-odd' or 'half-half' defining
        the partitioning rule. A user-defined partitioning can be defined by passing a tuple of the
        left and right indices. Defaults to `even-odd`.
    ordering
        The ordering with respect to which the partitioning rule is executed. Can be either
        'magnitude', 'random' or 'regular'. Defaults to 'regular'.
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

    generate_samples = staticmethod(_sample_transfer_function)

    def __init__(self, s, Hs, partitioning='even-odd', ordering='regular', force_real=True, mimo_handling='full'):
        assert isinstance(s, np.ndarray)
        assert partitioning in ('even-odd', 'half-half') \
            or len(partitioning) == 2 \
            and len(partitioning[0]) + len(partitioning[1]) == len(s)
        assert ordering in ('magnitude', 'random', 'regular')

        Hs = np.asarray(Hs)
        if Hs.ndim not in (1, 3):
            raise ValueError('Hs must contain sample data; use generate_samples to sample a model.')
        assert Hs.shape[0] == len(s)

        common_dtype = np.promote_types(s.dtype, Hs.dtype)
        Hs = Hs.astype(common_dtype, copy=False)

        # ensure that complex sampling values appear in complex conjugate pairs
        if force_real:
            old_s = s
            s, Hs = complete_conjugate_pairs(s, Hs)
            if isinstance(partitioning, tuple):
                p0, p1 = partitioning
                for i in range(len(old_s), len(s)):
                    source = np.flatnonzero(old_s == s[i].conj())[0]
                    if source in p0:
                        p0 = np.append(p0, i)
                    else:
                        p1 = np.append(p1, i)
                if len(p0) != len(partitioning[0]) or len(p1) != len(partitioning[1]):
                    self.logger.info('Added complex conjugates to partitionings. '
                                     f'New partitioning sizes are ({len(p0)}, {len(p1)}).')
                partitioning = (p0, p1)
            elif len(s) != len(old_s):
                self.logger.info(f'Added {len(s) - len(old_s)} complex conjugates to the data.')

        if len(Hs.shape) > 1:
            self.dim_output = Hs.shape[1]
            self.dim_input = Hs.shape[2]
            if self.dim_output == self.dim_input == 1:
                Hs = Hs[:, 0, 0]
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

        if self.force_real:
            A, B, C, E = A.real, B.real, C.real, E.real

        return LTIModel.from_matrices(A, B, C, D=None, E=E)


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
        ip, jp = partition_frequencies(self.s, self.Hs, self.partitioning, self.ordering, self.force_real)
        left_directions = right_directions = None
        if self.dim_input != 1 or self.dim_output != 1:
            if self.mimo_handling == 'random':
                rng = new_rng(0)
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
