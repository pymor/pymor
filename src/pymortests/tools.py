# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
from math import sin, pi, exp
import numpy as np
import pytest
import itertools
from tempfile import NamedTemporaryFile

from pymortests.base import TestInterface, runmodule
from pymortests.fixtures.grid import rect_or_tria_grid
from pymortests.base import polynomials
from pymor.tools.memory import total_size
from pymor.tools.quadratures import GaussQuadratures
from pymor.tools.floatcmp import float_cmp
from pymor.tools.vtkio import write_vtk
from pymor.la import NumpyVectorArray


class TestMemory(TestInterface):

    def test_positivity(self):
        for Class in [int, float, long, complex, str, unicode, list, tuple, bytearray, ]:
            r = Class()
            self.assertGreater(total_size(r), 0)

    def test_mem_growing(self):
        string = ''
        size = total_size(string)
        string = '*' * int(1e6)
        new_size = total_size(string)
        self.assertGreater(new_size, size)

    def test_custom_handler(self):
        class MyContainer(object):
            def items(self):
                for i in xrange(3):
                    yield i
        container = MyContainer()
        self.assertGreater(total_size(container, {MyContainer: MyContainer.items}), 0)


FUNCTIONS = (('sin(2x pi)', lambda x: sin(2 * x * pi), 0),
             ('e^x', lambda x: exp(x), exp(1) - exp(0)))


class TestGaussQuadrature(TestInterface):

    def test_polynomials(self):
        for n, function, _, integral in polynomials(GaussQuadratures.orders[-1]):
            name = 'x^{}'.format(n)
            for order in GaussQuadratures.orders:
                if n > (order) / 2:
                    continue
                Q = GaussQuadratures.iter_quadrature(order)
                ret = sum([function(p) * w for (p, w) in Q])
                self.assertAlmostEqual(ret, integral,
                                       msg='{} integral wrong: {} vs {} (quadrature order {})'
                                       .format(name, integral, ret, order))

    def test_other_functions(self):
        order = GaussQuadratures.orders[-1]
        for name, function, integral in FUNCTIONS:
            Q = GaussQuadratures.iter_quadrature(order)
            ret = sum([function(p) * w for (p, w) in Q])
            self.assertAlmostEqual(ret, integral,
                                   msg='{} integral wrong: {} vs {} (quadrature order {})'
                                   .format(name, integral, ret, order))

    def test_weights(self):
        for order in GaussQuadratures.orders:
            _, W = GaussQuadratures.quadrature(order)
            self.assertAlmostEqual(sum(W), 1)

    def test_points(self):
        for order in GaussQuadratures.orders:
            P, _ = GaussQuadratures.quadrature(order)
            np.testing.assert_array_equal(P, np.sort(P))
            self.assertLess(0.0, P[0])
            self.assertLess(P[-1], 1.0)


class TestCmp(TestInterface):

    def test_props(self):
        tol_range = [None, 0.0, 1]
        nan = float('nan')
        inf = float('inf')
        for (rtol, atol) in itertools.product(tol_range, tol_range):
            msg = 'rtol: {} | atol {}'.format(rtol, atol)
            self.assertTrue(float_cmp(0, 0, rtol, atol), msg)
            self.assertTrue(float_cmp(-0, -0, rtol, atol), msg)
            self.assertTrue(float_cmp(-1, -1, rtol, atol), msg)
            self.assertTrue(float_cmp(0, -0, rtol, atol), msg)
            self.assertFalse(float_cmp(2, -2, rtol, atol), msg)

            self.assertFalse(float_cmp(nan, nan, rtol, atol), msg)
            self.assertTrue(nan != nan)
            self.assertFalse(nan == nan)
            self.assertFalse(float_cmp(-nan, nan, rtol, atol), msg)

            self.assertFalse(float_cmp(inf, inf, rtol, atol), msg)
            self.assertFalse(inf != inf)
            self.assertTrue(inf == inf)
            self.assertTrue(float_cmp(-inf, inf, rtol, atol), msg)


def test_vtkio(rect_or_tria_grid):
    grid = rect_or_tria_grid
    steps = 4
    for dim in range(1, 2):
        for codim, data in enumerate((NumpyVectorArray(np.zeros((steps, grid.size(c)))) for c in range(grid.dim+1))):
            with NamedTemporaryFile('wb') as out:
                if codim == 1:
                    with pytest.raises(NotImplementedError):
                        write_vtk(grid, data, out.name, codim=codim)
                else:
                    write_vtk(grid, data, out.name, codim=codim)

if __name__ == "__main__":
    runmodule(filename=__file__)
