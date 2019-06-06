# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.interfaces import BasicInterface
from pymor.models.iosys import SecondOrderModel
from pymor.reductors.interpolation import SOBHIReductor
from pymor.reductors.h2 import (
    IRKAReductor,
    _poles_and_tangential_directions,
    _convergence_criterion,
)


class SORIRKAReductor(BasicInterface):
    """SOR-IRKA reductor.

    Parameters
    ----------
    fom
        The full-order |SecondOrderModel| to reduce.
    """

    def __init__(self, fom):
        assert isinstance(fom, SecondOrderModel)
        self.fom = fom
        self.V = None
        self.W = None
        self._pg_reductor = None
        self.conv_crit = None
        self.sigmas = None
        self.R = None
        self.L = None
        self.errors = None

    def reduce(
        self,
        r,
        sigma=None,
        b=None,
        c=None,
        rom0=None,
        tol=1e-4,
        maxit=100,
        num_prev=1,
        force_sigma_in_rhp=False,
        projection="orth",
        conv_crit="sigma",
        compute_errors=False,
        irka_options=None,
    ):
        r"""Reduce using SOR-IRKA.

        It uses IRKA as the intermediate reductor, to reduce from 2r to
        r poles.
        See Section 5.3.2 in [W12]_.

        Parameters
        ----------
        r
            Order of the reduced order model.
        sigma
            Initial interpolation points (closed under conjugation).

            If `None`, interpolation points are log-spaced between 0.1
            and 10. If `sigma` is an `int`, it is used as a seed to
            generate it randomly. Otherwise, it needs to be a
            one-dimensional array-like of length `r`.

            `sigma` and `rom0` cannot both be not `None`.
        b
            Initial right tangential directions.

            If `None`, if is chosen as all ones. If `b` is an `int`, it
            is used as a seed to generate it randomly. Otherwise, it
            needs to be a |VectorArray| of length `r` from `fom.B.source`.

            `b` and `rom0` cannot both be not `None`.
        c
            Initial left tangential directions.

            If `None`, if is chosen as all ones. If `c` is an `int`, it
            is used as a seed to generate it randomly. Otherwise, it
            needs to be a |VectorArray| of length `r` from `fom.Cp.range`.

            `c` and `rom0` cannot both be not `None`.
        rom0
            Initial reduced order model.

            If `None`, then `sigma`, `b`, and `c` are used. Otherwise,
            it needs to be an |LTIModel| of order `r` and it is used to
            construct `sigma`, `b`, and `c`.
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
        irka_options
            Dict of options for IRKAReductor.reduce.

        Returns
        -------
        rom
            Reduced-order |SecondOrderModel|.
        """
        fom = self.fom
        if not fom.cont_time:
            raise NotImplementedError
        assert 0 < r < fom.order
        assert isinstance(num_prev, int) and num_prev >= 1
        assert projection in ("orth", "biorth")
        assert conv_crit in ("sigma", "h2")
        assert irka_options is None or isinstance(irka_options, dict)
        if not irka_options:
            irka_options = {}

        # initial interpolation points and tangential directions
        assert sigma is None or isinstance(sigma, int) or len(sigma) == r
        assert b is None or isinstance(b, int) or b in fom.B.source and len(b) == r
        assert c is None or isinstance(c, int) or c in fom.Cp.range and len(c) == r
        assert (
            rom0 is None
            or isinstance(rom0, SecondOrderModel)
            and rom0.order == r
            and rom0.B.source == fom.B.source
            and rom0.Cp.range == fom.Cp.range
        )
        assert sigma is None or rom0 is None
        assert b is None or rom0 is None
        assert c is None or rom0 is None
        if rom0 is not None:
            with self.logger.block("Intermediate reduction ..."):
                irka_reductor = IRKAReductor(rom0.to_lti())
                rom_r = irka_reductor.reduce(r, **irka_options)
            poles, b, c = _poles_and_tangential_directions(rom_r)
            sigma = (
                np.abs(poles.real) + poles.imag * 1j if force_sigma_in_rhp else -poles
            )
        else:
            if sigma is None:
                sigma = np.logspace(-1, 1, r)
            elif isinstance(sigma, int):
                np.random.seed(sigma)
                sigma = np.abs(np.random.randn(r))
            if b is None:
                b = fom.B.source.ones(r)
            elif isinstance(b, int):
                b = fom.B.source.random(r, distribution="normal", seed=b)
            if c is None:
                c = fom.Cp.range.ones(r)
            elif isinstance(c, int):
                c = fom.Cp.range.random(r, distribution="normal", seed=c)

        self.logger.info("Starting SOR-IRKA")
        self.conv_crit = []
        self.sigmas = [np.array(sigma)]
        self.R = [b]
        self.L = [c]
        self.errors = [] if compute_errors else None
        self._pg_reductor = SOBHIReductor(fom)
        # main loop
        for it in range(maxit):
            # interpolatory reduced order model
            rom = self._pg_reductor.reduce(sigma, b, c, projection=projection)

            # reduction to a system with r poles
            with self.logger.block("Intermediate reduction ..."):
                irka_reductor = IRKAReductor(rom.to_lti())
                rom_r = irka_reductor.reduce(r, **irka_options)

            # new interpolation points and tangential directions
            poles, b, c = _poles_and_tangential_directions(rom_r)
            sigma = (
                np.abs(poles.real) + poles.imag * 1j if force_sigma_in_rhp else -poles
            )
            self.sigmas.append(sigma)
            self.R.append(b)
            self.L.append(c)

            # compute convergence criterion
            if conv_crit == "sigma":
                dist = _convergence_criterion(
                    self.sigmas[: -num_prev - 2 : -1], conv_crit
                )
                self.conv_crit.append(dist)
            elif conv_crit == "h2":
                if it == 0:
                    rom_list = (num_prev + 1) * [None]
                    rom_list[0] = rom
                    self.conv_crit.append(np.inf)
                else:
                    rom_list[1:] = rom_list[:-1]
                    rom_list[0] = rom
                    dist = _convergence_criterion(rom_list, conv_crit)
                    self.conv_crit.append(dist)

            # report convergence
            self.logger.info(
                f"Convergence criterion in iteration {it + 1}: {self.conv_crit[-1]:e}"
            )
            if compute_errors:
                if np.max(rom.poles().real) < 0:
                    err = fom - rom
                    rel_H2_err = err.h2_norm() / fom.h2_norm()
                else:
                    rel_H2_err = np.inf
                self.errors.append(rel_H2_err)

                self.logger.info(
                    f"Relative H2-error in iteration {it + 1}: {rel_H2_err:e}"
                )

            # check if convergence criterion is satisfied
            if self.conv_crit[-1] < tol:
                break

        # final reduced order model
        rom = self._pg_reductor.reduce(sigma, b, c, projection=projection)
        self.V = self._pg_reductor.V
        self.W = self._pg_reductor.W
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)
