from pymortests.base import runmodule
import numpy as np
from pymor.core.config import config
from pymor.analyticalproblems.expressions import parse_expression
import os
import pytest


def test_fenics_expression_vectorized():
    try:
        from dolfin import UnitSquareMesh
    except ImportError as ie:
        if not os.environ.get('DOCKER_PYMOR', False):
            pytest.skip('skipped test due to missing fenics')
        raise ie
    e = parse_expression('[1., 2.] + [x[0], x[1]]', {'x': 2})
    mesh = UnitSquareMesh(10,10)
    f_fenics, f_fenics_params = e.to_fenics(mesh)
    eval = f_fenics.item()([5., 2.])
    val = [1. + 5., 2. + 2.]

    assert np.allclose(eval, val)


def test_fenics_expression_vectorized():
    try:
        from dolfin import UnitSquareMesh
    except ImportError as ie:
        if not os.environ.get('DOCKER_PYMOR', False):
            pytest.skip('skipped test due to missing fenics')
        raise ie
    e = parse_expression('sin([x[0], x[1]])', {'x': 2})
    mesh = UnitSquareMesh(10,10)
    f_fenics, f_fenics_params = e.to_fenics(mesh)
    eval = f_fenics.item()([5., 2.])
    val = [np.sin(5.), np.sin(2.)]

    assert np.allclose(eval, val)


def test_fenics_expression_scalar():
    try:
        from dolfin import UnitSquareMesh
    except ImportError as ie:
        if not os.environ.get('DOCKER_PYMOR', False):
            pytest.skip('skipped test due to missing fenics')
        raise ie
    e = parse_expression('x[0] + sin(x[1])', {'x': 2})
    mesh = UnitSquareMesh(10,10)
    f_fenics, f_fenics_params = e.to_fenics(mesh)
    eval = f_fenics.item()([5., 2.])
    val = 5. + np.sin(2.)

    assert np.allclose(eval, val)


if __name__ == "__main__":
    runmodule(filename=__file__)
