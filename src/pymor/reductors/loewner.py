# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.loewner import loewner_quadruple
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
          directions. If `conjugate=True`, directions at conjugate nodes must be conjugates.
    """

    cache_region = 'memory'

    def __init__(self, s, Hs, partitioning='even-odd', ordering='regular', conjugate=True, mimo_handling='full'):
        assert isinstance(s, np.ndarray)
        if hasattr(Hs, 'transfer_function'):
            Hs = Hs.transfer_function
        assert isinstance(Hs, TransferFunction | np.ndarray | list)

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
            assert Hs.shape[0] == len(s)

        common_dtype = np.promote_types(s.dtype, Hs.dtype)
        Hs = Hs.astype(common_dtype, copy=False)

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
            if self.dim_output == self.dim_input == 1:
                Hs = np.squeeze(Hs)
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
                idx = np.argsort([np.linalg.norm(self.Hs[i]) for i in range(len(self.Hs))])
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
        r"""Construct a Loewner quadruple as |NumPy arrays|.

        The Loewner quadruple :cite:`ALI17`

        .. math::
            (\mathbb{L},\mathbb{L}_s,V,W)

        consists of the Loewner matrix :math:`\mathbb{L}`, the shifted Loewner matrix
        :math:`\mathbb{L}_s`, left interpolation data :math:`V` and right interpolation
        data :math:`W`.
        """
        ip, jp = self._partition_frequencies() if isinstance(self.partitioning, str) else self.partitioning
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
            left_directions=left_directions, right_directions=right_directions, force_real=self.conjugate,
        )

    @cached
    def _loewner_svds(self, L, Ls):
        """Compute SVDs of stacked Loewner matrices."""
        LhLs = np.hstack([L, Ls])
        Y, S1, _ = spla.svd(LhLs, full_matrices=False)
        LvLs = np.vstack([L, Ls])
        _, S2, Xh = spla.svd(LvLs, full_matrices=False)

        return Y, S1, S2, Xh
