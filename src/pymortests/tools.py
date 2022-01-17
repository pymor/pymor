# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import itertools
import os
import tempfile
from math import exp, factorial, pi, sin
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given

from pymor.core.config import config
from pymor.core.logger import getLogger
from pymor.discretizers.builtin.quadratures import GaussQuadratures
from pymor.tools import formatsrc, timing
from pymor.tools.deprecated import Deprecated
from pymor.tools.floatcmp import almost_less, float_cmp, float_cmp_all
from pymor.tools.formatsrc import print_source
from pymor.tools.io import safe_temporary_filename, change_to_directory
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.base import runmodule
from pymortests.fixtures.grid import hy_rect_or_tria_grid

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


def test_almost_less():
    tol_range = [0.0, 1e-8, 1]
    inf = float('inf')
    for (rtol, atol) in itertools.product(tol_range, tol_range):
        msg = f'rtol: {rtol} | atol {atol}'
        assert almost_less(0., 1, rtol, atol), msg
        assert almost_less(-1., -0., rtol, atol), msg
        assert almost_less(-1, -0.9, rtol, atol), msg
        assert almost_less(0., 1, rtol, atol), msg
        assert almost_less(-1., 1., rtol, atol), msg
        assert almost_less(atol, 0., rtol, atol), msg
        assert (rtol == 0.0 and not almost_less(0., inf, rtol, atol), msg) or \
               almost_less(0., inf, rtol, atol), msg


@pytest.mark.skipif(not config.HAVE_VTKIO, reason='VTKIO support libraries missing')
@given(hy_rect_or_tria_grid)
def test_vtkio(grid):
    from pymor.discretizers.builtin.grids.vtkio import write_vtk
    from pymor.tools.io.vtk import read_vtkfile
    steps = 4
    for codim, data in enumerate((NumpyVectorSpace.from_numpy(np.ones((steps, grid.size(c))))
                                  for c in range(grid.dim+1))):
        with safe_temporary_filename('wb') as out_name:
            fn = write_vtk(grid, data, out_name, codim=codim)
            meshes = read_vtkfile(fn)
            assert len(meshes) == len(data)
            assert all((a is not None and b is not None for a, b in meshes))


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


def test_formatsrc():
    obj = formatsrc.format_source
    formatsrc.format_source(obj)
    print_source(obj)


def test_formatsrc_nopygments(monkeypatch):
    try:
        from pygments import highlight  # noqa: F401
        monkeypatch.delattr('pygments.highlight')
    except ImportError:
        pass
    test_formatsrc()


def test_load_matrix(loadable_matrices):
    from pymor.tools.io import load_matrix
    for m in loadable_matrices:
        if m.suffix == '.npz':
            load_matrix(m, key='arr_0')
        else:
            load_matrix(m)


@pytest.mark.parametrize('ext', ['.mat', '.mtx', '.mtz.gz', '.npy', '.npz', '.txt'])
def test_save_load_matrix(ext):
    import filecmp
    from pymor.tools.io import load_matrix, save_matrix
    A = np.eye(2)
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, 'matrix' + ext)
        key = None
        if ext == '.mat':
            key = 'A'
        save_matrix(path, A, key)
        B = load_matrix(path, key)
        assert np.all(A == B)
        path2 = os.path.join(tmpdirname, 'matrix2' + ext)
        save_matrix(path2, A, key)
        # .mat save a timestamp, so full file cmp os too flaky
        if ext not in ('.mtz.gz', '.mat'):
            assert filecmp.cmp(path, path2)


def test_cwd_ctx_manager():
    def _cwd():
        return Path(os.getcwd()).resolve()

    original_cwd = _cwd()
    target = Path(tempfile.gettempdir()).resolve()
    with change_to_directory(target) as result:
        assert result is None
        assert _cwd() == target
    assert _cwd() == original_cwd


if __name__ == "__main__":
    runmodule(filename=__file__)
