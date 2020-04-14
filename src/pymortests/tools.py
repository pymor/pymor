# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from math import sin, pi, exp, factorial
import numpy as np
import pytest
import itertools

from hypothesis import given

from pymor.core.logger import getLogger
from pymor.tools.io import SafeTemporaryFileName
from pymortests.base import runmodule
from pymortests.fixtures.grid import hy_rect_or_tria_grid
from pymor.discretizers.builtin.grids.vtkio import write_vtk
from pymor.discretizers.builtin.quadratures import GaussQuadratures
from pymor.tools.deprecated import Deprecated
from pymor.tools.floatcmp import float_cmp, float_cmp_all
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.tools import timing


logger = getLogger('pymortests.tools')


FUNCTIONS = (('sin(2x pi)', lambda x: sin(2 * x * pi), 0),
             ('e^x', lambda x: exp(x), exp(1) - exp(0)))


def polynomials(max_order):
    for n in range(max_order + 1):
        def f(x):
            return np.power(x, n)

        def deri(k):
            if k > n:
                return lambda _: 0
            return lambda x: (factorial(n) / factorial(n - k)) * np.power(x, n - k)

        integral = (1 / (n + 1))
        yield (n, f, deri, integral)


def test_quadrature_polynomials():
    for n, function, _, integral in polynomials(GaussQuadratures.orders[-1]):
        name = f'x^{n}'
        for order in GaussQuadratures.orders:
            if n > order / 2:
                continue
            Q = GaussQuadratures.iter_quadrature(order)
            ret = sum([function(p) * w for (p, w) in Q])
            assert float_cmp(ret, integral), \
                f'{name} integral wrong: {integral} vs {ret} (quadrature order {order})'


def test_quadrature_other_functions():
    order = GaussQuadratures.orders[-1]
    for name, function, integral in FUNCTIONS:
        Q = GaussQuadratures.iter_quadrature(order)
        ret = sum([function(p) * w for (p, w) in Q])
        assert float_cmp(ret, integral), \
            f'{name} integral wrong: {integral} vs {ret} (quadrature order {order})'


def test_quadrature_weights():
    for order in GaussQuadratures.orders:
        _, W = GaussQuadratures.quadrature(order)
        assert float_cmp(sum(W), 1)


def test_quadrature_points():
    for order in GaussQuadratures.orders:
        P, _ = GaussQuadratures.quadrature(order)
        assert float_cmp_all(P, np.sort(P))
        assert 0.0 < P[0]
        assert P[-1] < 1.0


def test_float_cmp():
    tol_range = [0.0, 1e-8, 1]
    nan = float('nan')
    inf = float('inf')
    for (rtol, atol) in itertools.product(tol_range, tol_range):
        msg = f'rtol: {rtol} | atol {atol}'
        assert float_cmp(0., 0., rtol, atol), msg
        assert float_cmp(-0., -0., rtol, atol), msg
        assert float_cmp(-1., -1., rtol, atol), msg
        assert float_cmp(0., -0., rtol, atol), msg
        assert not float_cmp(2., -2., rtol, atol), msg

        assert not float_cmp(nan, nan, rtol, atol), msg
        assert nan != nan
        assert not (nan == nan)
        assert not float_cmp(-nan, nan, rtol, atol), msg

        assert not float_cmp(inf, inf, rtol, atol), msg
        assert not (inf != inf)
        assert inf == inf
        if rtol > 0:
            assert float_cmp(-inf, inf, rtol, atol), msg
        else:
            assert not float_cmp(-inf, inf, rtol, atol), msg


@given(hy_rect_or_tria_grid)
def test_vtkio(grid):
    steps = 4
    for dim in range(1, 2):
        for codim, data in enumerate((NumpyVectorSpace.from_numpy(np.zeros((steps, grid.size(c))))
                                      for c in range(grid.dim+1))):
            with SafeTemporaryFileName('wb') as out_name:
                if codim == 1:
                    with pytest.raises(NotImplementedError):
                        write_vtk(grid, data, out_name, codim=codim)
                else:
                    write_vtk(grid, data, out_name, codim=codim)


def testTimingContext():
    with timing.Timer('busywait', logger):
        timing.busywait(100)
    with timing.Timer('defaultlog'):
        timing.busywait(100)


def testTimingDecorator():

    @timing.Timer('busywait_decorator', logger)
    def wait():
        timing.busywait(1000)

    wait()


def testTiming():
    timer = timing.Timer('busywait', logger)
    timer.start()
    timing.busywait(1000)
    timer.stop()
    logger.info('plain timing took %s seconds', timer.dt)


def testDeprecated():
    @Deprecated('use other stuff instead')
    def deprecated_function():
        pass
    # Cause all warnings to always be triggered.
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Trigger a warning.
        deprecated_function()
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "DeprecationWarning" in str(w[-1].message)


if __name__ == "__main__":
    runmodule(filename=__file__)
