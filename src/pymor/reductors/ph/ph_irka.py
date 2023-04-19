# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.models.iosys import PHLTIModel
from pymor.reductors.h2 import GenericIRKAReductor, OneSidedIRKAReductor
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

    def reduce(self, rom0_params, tol=1e-4, maxit=100, num_prev=1, projection='orth', conv_crit='sigma',
               compute_errors=False):
        r"""Reduce using pH-IRKA.

        It uses IRKA as the intermediate reductor.

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

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product.
            - `'Eorth'`: projection matrix is orthogonalized with
              respect to the E product.
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
        one_sided_irka_reductor = OneSidedIRKAReductor(self.fom.to_lti(), 'V')
        _ = one_sided_irka_reductor.reduce(rom0_params, tol=tol, maxit=maxit, num_prev=num_prev,
                                           projection=projection, conv_crit=conv_crit, compute_errors=compute_errors)

        self._pg_reductor = PHLTIPGReductor(self.fom, one_sided_irka_reductor.V)
        rom = self._pg_reductor.reduce()

        return rom
