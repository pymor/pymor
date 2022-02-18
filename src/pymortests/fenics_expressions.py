from pymortests.base import runmodule
import numpy as np
from pymor.core.config import config
from pymor.analyticalproblems.expressions import parse_expression


def test_fenics_expression_vectorized():
    from dolfin import UnitSquareMesh
    e = parse_expression('sin([x[0], x[1]])', {'x': 2})
    mesh = UnitSquareMesh(10,10)
    f_fenics, f_fenics_params = e.to_fenics(mesh)
    eval = f_fenics.item()([5., 2.])
    val = [np.sin(5.), np.sin(2.)]

    assert np.allclose(eval, val)


def test_fenics_expression_scalar():
    from dolfin import UnitSquareMesh
    e = parse_expression('x[0] + sin(x[1])', {'x': 2})
    mesh = UnitSquareMesh(10,10)
    f_fenics, f_fenics_params = e.to_fenics(mesh)
    eval = f_fenics.item()([5., 2.])
    val = 5. + np.sin(2.)

    assert np.allclose(eval, val)


if __name__ == "__main__":
    if config.HAVE_FENICS: # this just does not need to run without fenics
        runmodule(filename=__file__)
