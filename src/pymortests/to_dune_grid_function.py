# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from dune.xt.functions import ConstantFunction as DuneConstantFunction, GridFunction as GF
from dune.xt.grid import Dim, Cube, make_cube_grid

from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction, ProductFunction, ExpressionFunction
from pymor.discretizers.dunegdt.functions import to_dune_function, to_dune_grid_function
from pymor.parameters.functionals import ExpressionParameterFunctional

grid = make_cube_grid(Dim(2), Cube(), [0, 0], [1, 1], [1, 1])


def test_ConstTantFunction_to_dune_function():
    from dune.xt.functions import ConstantFunction2To1d

    func = to_dune_function(ConstantFunction(value=1, dim_domain=2))

    assert isinstance(func, ConstantFunction2To1d)
    assert func.evaluate([0, 0])[0] == 1


def tets_ProductFunction_to_dune_function():
    from dune.xt.functions import ProductFunction__2d_to_1x1

    func = to_dune_function(ConstantFunction(value=1, dim_domain=2) \
                            * ConstantFunction(value=2, dim_domain=2) \
                            * ConstantFunction(value=3, dim_domain=2))

    assert isinstance(func, ProductFunction__2d_to_1x1)
    assert func.evaluate([0, 0])[0] == 6


def test_LincombFunction_to_dune_function():
    from dune.xt.functions import ConstantFunction2To1d
    f = ConstantFunction(value=1, dim_domain=2)

    [funcs, coeffs] = to_dune_function(f + 2*f + 3*f)
    assert all([isinstance(fun, ConstantFunction2To1d) for fun in funcs])
    assert np.allclose(coeffs, [1, 2, 3])


def test_convertible_function_to_dune_grid_function():
    from dune.xt.functions import GridFunction2dCubeYaspgridTo1d

    f = ConstantFunction(value=1, dim_domain=2)
    for func in (f, f*f*f):
        func = to_dune_grid_function(f, grid)
        assert isinstance(func, GridFunction2dCubeYaspgridTo1d)


def test_ExpressionFunction_to_dune_grid_function():
    f = ExpressionFunction('x[..., 0]', dim_domain=2)
    df = to_dune_grid_function(f, grid)

    assert np.allclose(f.evaluate(grid.centers(grid.dimension)),
                       np.array(df.dofs.vector, copy=False))


def test_parametric_ExpressionFunction_to_dune_grid_function():
    f = ExpressionFunction('foo[0]*x[..., 0]', dim_domain=2, parameters={'foo': 1})
    mu = f.parameters.parse(2)
    df = to_dune_grid_function(f, grid, mu=mu)

    assert np.allclose(f.evaluate(grid.centers(grid.dimension), mu=mu),
                       np.array(df.dofs.vector, copy=False))


def test_vector_valued_ExpressionFunction_to_dune_grid_function():
    f = ExpressionFunction('np.array([x[..., 0], x[..., 1]]).T', dim_domain=2, shape_range=(2,))
    df = to_dune_grid_function(f, grid)

    f_vals = f.evaluate(grid.centers(grid.dimension))
    f_vals = np.hstack([f_vals[:, 0], f_vals[:, 1]])

    assert np.allclose(f_vals,
                       np.array(df.dofs.vector, copy=False))
