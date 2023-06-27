# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.krylov import tangential_rational_krylov
from pymor.models.iosys import PHLTIModel
from pymor.reductors.h2 import GenericIRKAReductor
from pymor.reductors.ph.basic import PHLTIPGReductor


class PHIRKAReductor(GenericIRKAReductor):
    """PH-IRKA reductor.

    Parameters
    ----------
    fom
        The full-order |PHLTIModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, PHLTIModel)
        super().__init__(fom, mu=mu)

    def reduce(self, rom0_params, tol=1e-4, maxit=100, num_prev=1,
               projection='orth', conv_crit='sigma',
               compute_errors=False):
        r"""Reduce using pH-IRKA.

        See :cite:`GPBV12`.

        Parameters
        ----------
        rom0_params
            Can be:

            - order of the reduced model (a positive integer),
            - initial interpolation points (a 1D |NumPy array|),
            - dict with `'sigma'`, `'b'`, `'c'` as keys mapping to
              initial interpolation points (a 1D |NumPy array|), right
              tangential directions (|NumPy array| of shape
              `(len(sigma), fom.dim_input)`), and left tangential directions
              (|NumPy array| of shape `(len(sigma), fom.dim_input)`),
            - initial reduced-order model (|LTIModel|).

            If the order of reduced model is given, initial
            interpolation data is generated randomly.
        tol
            Tolerance for the convergence criterion.
        maxit
            Maximum number of iterations.
        num_prev
            Number of previous iterations to compare the current
            iteration to. Larger number can avoid occasional cyclic
            behavior of IRKA.
        projection
            Projection method:

            - `'orth'`: projection matrix `V` is orthogonalized with
              respect to the Euclidean inner product.
            - `'QTEorth'`: projection matrix `V` is orthogonalized with
              respect to the `fom.Q.H @ fom.E` product.
        conv_crit
            Convergence criterion:

            - `'sigma'`: relative change in interpolation points
            - `'h2'`: relative :math:`\mathcal{H}_2` distance of
              reduced-order models
        compute_errors
            Should the relative :math:`\mathcal{H}_2`-errors of
            intermediate reduced order models be computed.

            .. warning::
                Computing :math:`\mathcal{H}_2`-errors is expensive. Use
                this option only if necessary.

        Returns
        -------
        rom
            Reduced-order |PHLTIModel|.
        """
        if self.fom.sampling_time > 0:
            raise NotImplementedError

        self._clear_lists()
        sigma, b, c = self._rom0_params_to_sigma_b_c(rom0_params, False)
        self._store_sigma_b_c(sigma, b, c)
        self._check_common_args(tol, maxit, num_prev, conv_crit)
        assert projection in ('orth', 'QTEorth')

        self.logger.info('Starting pH-IRKA')
        self._conv_data = (num_prev + 1) * [None]
        if conv_crit == 'sigma':
            self._conv_data[0] = sigma
        for it in range(maxit):
            self._set_V_reductor(sigma, b, projection)
            rom = self._pg_reductor.reduce()
            sigma, b, c = self._rom_to_sigma_b_c(rom, False)
            self._store_sigma_b_c(sigma, b, c)
            self._update_conv_data(sigma, rom, conv_crit)
            self._compute_conv_crit(rom, conv_crit, it)
            self._compute_error(rom, it, compute_errors)
            if self.conv_crit[-1] < tol:
                break

        return rom

    def _assemble_fom(self):
        return (
            self.fom.with_(
                **{op: getattr(self.fom, op).assemble(mu=self.mu)
                   for op in 'JRGPSNEQ'}
            )
            if self.fom.parametric
            else self.fom
        )

    def _set_V_reductor(self, sigma, b, projection):
        fom = self._assemble_fom()
        self.V = tangential_rational_krylov(fom.A, fom.E, fom.B, fom.B.source.from_numpy(b), sigma, orth=False)
        product = None if projection == 'orth' else fom.Q.H @ fom.E
        gram_schmidt(self.V, atol=0, rtol=0, product=product, copy=False)
        self._pg_reductor = PHLTIPGReductor(fom, self.V, projection == 'QTEorth')
        self.W = self._pg_reductor.bases['W']
