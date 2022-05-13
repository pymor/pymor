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

    def reduce(self, rom0_params, tol=1e-4, maxit=100, num_prev=1,
               force_sigma_in_rhp=False, projection='orth', conv_crit='sigma',
               compute_errors=False):
        r"""Reduce using PH-IRKA.

        It uses IRKA as the intermediate reductor.

        Parameters
        ----------
        rom0_params
            Can be:

            - order of the reduced model (a positive integer),
            - dict with `'sigma'`, `'b'`, `'c'` as keys mapping to
              initial interpolation points (a 1D |NumPy array|), right
              tangential directions (|VectorArray| from
              `fom.D.source`), and left tangential directions
              (|VectorArray| from `fom.D.range`), all of the same
              length (the order of the reduced model),
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
            If `False`, new interpolation are reflections of the current
            reduced order model's poles. Otherwise, only the poles in
            the left half-plane are reflected.
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogolized with
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

        Returns
        -------
        rom
            Reduced-order |PHLTIModel|.
        """
        one_sided_irka_reductor = OneSidedIRKAReductor(self.fom.to_lti(), 'V')
        _ = one_sided_irka_reductor.reduce(rom0_params, tol, maxit, num_prev,
                                           force_sigma_in_rhp, projection, conv_crit,
                                           compute_errors)

        self._pg_reductor = PHLTIPGReductor(self.fom, one_sided_irka_reductor.V)
        rom = self._pg_reductor.reduce()

        return rom
