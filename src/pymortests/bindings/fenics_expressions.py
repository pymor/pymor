from pymortests.base import runmodule
import numpy as np
from pymor.analyticalproblems.expressions import parse_expression
from pymortests.base import skip_if_missing


@skip_if_missing('FENICS')
def test_fenics_expression_vectorized():
    from dolfin import UnitSquareMesh
    e = parse_expression('[1., 2.] + [x[0], x[1]]', {'x': 2})
    mesh = UnitSquareMesh(10, 10)
    f_fenics, f_fenics_params = e.to_fenics(mesh)
    eval = [f_fenics[0]([5., 2.]), f_fenics[1]([5., 2.])]
    val = [1. + 5., 2. + 2.]

    assert np.allclose(eval, val)


@skip_if_missing('FENICS')
def test_fenics_expression_vectorized2():
    from dolfin import UnitSquareMesh
    e = parse_expression('sin([x[0], x[1]])', {'x': 2})
    mesh = UnitSquareMesh(10, 10)
    f_fenics, f_fenics_params = e.to_fenics(mesh)
    eval = [f_fenics[0]([5., 2.]), f_fenics[1]([5., 2.])]
    val = [np.sin(5.), np.sin(2.)]

    assert np.allclose(eval, val)


@skip_if_missing('FENICS')
def test_fenics_expression_scalar():
    from dolfin import UnitSquareMesh
    e = parse_expression('x[0] + sin(x[1])', {'x': 2})
    mesh = UnitSquareMesh(10, 10)
    f_fenics, f_fenics_params = e.to_fenics(mesh)
    eval = f_fenics.item()([5., 2.])
    val = 5. + np.sin(2.)

    assert np.allclose(eval, val)


if __name__ == "__main__":
    runmodule(filename=__file__)
