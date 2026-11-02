# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""IRKA-type reductor for |SecondOrderModels|."""

from functools import partial

from pymor.models.iosys import SecondOrderModel
from pymor.reductors.h2 import GenericIRKAReductor, IRKAReductor
from pymor.reductors.interpolation import SOBHIReductor


class SORIRKAReductor(GenericIRKAReductor):
    """SOR-IRKA reductor.

    Parameters
    ----------
    fom
        The full-order |SecondOrderModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, SecondOrderModel)
        super().__init__(fom, mu=mu)

    def reduce(self, rom0_params, tol=1e-4, maxit=100, num_prev=1,
               force_sigma_in_rhp=False, projection='orth', conv_crit='sigma',
               compute_errors=False, irka_options=None):
        r"""Reduce using SOR-IRKA.

        It uses IRKA as the intermediate reductor, to reduce from 2r to
        r poles. See Section 5.3.2 in :cite:`W12`.

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
              (|NumPy array| of shape `(len(sigma), fom.dim_output)`),
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
        force_sigma_in_rhp
            If `False`, new interpolation points are reflections of the current
            reduced order model's poles. Otherwise, only the poles in
            the left half-plane are reflected.
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogonalized with
              respect to the E product
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
        irka_options
            Dict of options for IRKAReductor.reduce.

        Returns
        -------
        rom
            Reduced-order |SecondOrderModel|.
        """
        assert projection in ('orth', 'biorth')
        assert irka_options is None or isinstance(irka_options, dict)
        if not irka_options:
            irka_options = {}

        rom_to_sigma_b_c = partial(self._rom_to_sigma_b_c, force_sigma_in_rhp=force_sigma_in_rhp)

        def update(rom):
            with self.logger.block('Intermediate reduction ...'):
                irka_reductor = IRKAReductor(rom.to_lti())
                rom_r = irka_reductor.reduce(rom.order, **irka_options)
            return rom_to_sigma_b_c(rom_r)

        data = self._rom0_params_to_sigma_b_c(rom0_params, rom_to_sigma_b_c)
        step = partial(self._bhi_step, SOBHIReductor(self.fom, mu=self.mu), projection=projection)
        return self._iterate('SOR-IRKA', data, step, update,
                             tol, maxit, num_prev, conv_crit, compute_errors)
