# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from dune.xt.functions import (
        ConstantFunction as DuneConstantFunction,
        GridFunction as GF,
        ConstantFunction2To1d,
        GridFunction2dCubeYaspgridTo1d)
from dune.xt.grid import Dim, Cube, make_cube_grid

from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction, ProductFunction, ExpressionFunction
from pymor.discretizers.dunegdt.functions import (to_dune_function, to_dune_grid_function, DuneFunction,
                                                  DuneGridFunction, LincombDuneFunction, LincombDuneGridFunction)
from pymor.parameters.functionals import ExpressionParameterFunctional

grid = make_cube_grid(Dim(2), Cube(), [0, 0], [1, 1], [1, 1])


def _make_dune_func(value, as_='', grid=None):
    func = ConstantFunction2To1d([value,])
    if grid is not None:
        func = GF(grid, func)
    if as_ == 'DuneFunction':
        if grid is None:
            return DuneFunction(func)
        else:
            return DuneGridFunction(func)
    elif as_ == 'LincombDuneFunction':
        if grid is None:
            return LincombDuneFunction([func,], [1,])
        else:
            return LincombDuneGridFunction([func,], [1,])
    else:
        return func


def test_LincombDuneFunction_to_dune_function():
    func = _make_dune_func(0.5, 'LincombDuneFunction')
    dune_func = to_dune_function(func, ensure_lincomb=True)
    assert dune_func == func
    dune_func = to_dune_function(func, ensure_lincomb=False)
    assert dune_func == func

def test_DuneFunction_to_dune_function():
    func = _make_dune_func(0.5, 'DuneFunction')
    dune_func = to_dune_function(func, ensure_lincomb=False)
    assert dune_func == func
    dune_func = to_dune_function(func, ensure_lincomb=True)
    assert isinstance(dune_func, LincombDuneFunction)
    assert dune_func.functions[0] == func.impl

def test_ConstTantFunction_to_dune_function():
    pymor_func = ConstantFunction(value=42.17, dim_domain=2)
    func = to_dune_function(pymor_func)
    assert isinstance(func, DuneFunction)
    lincomb_func = to_dune_function(pymor_func, ensure_lincomb=True)
    assert isinstance(lincomb_func, LincombDuneFunction)
    for f in func.impl, lincomb_func.functions[0]:
        assert isinstance(f, ConstantFunction2To1d)
        assert f.evaluate([0, 0])[0] == 42.17

def tets_ProductFunction_to_dune_function():
    from dune.xt.functions import ProductFunction__2d_to_1x1

    pymor_func = ConstantFunction(value=1, dim_domain=2) \
                 * ConstantFunction(value=2, dim_domain=2) \
                 * ConstantFunction(value=3, dim_domain=2)

    func = to_dune_function(pymor_func)
    assert isinstance(func, DuneFunction)

    lincomb_func = to_dune_function(pymor_func, ensure_lincomb=True)
    assert isinstance(lincomb_func, LincombDuneFunction)

    for f in func.impl, lincomb_func.functions[0]:
        assert isinstance(func, ProductFunction__2d_to_1x1)
        assert func.evaluate([0, 0])[0] == 6

def test_LincombFunction_to_dune_function():
    f = ConstantFunction(value=1, dim_domain=2)
    pymor_func = f + 2*f + 3*f

    lincomb_func1 = to_dune_function(pymor_func)
    lincomb_func2 = to_dune_function(pymor_func, ensure_lincomb=True)

    for f in lincomb_func1, lincomb_func2:
        assert isinstance(f, LincombDuneFunction)
        assert all(isinstance(func, ConstantFunction2To1d) for func in f.functions)
        assert np.allclose(f.coefficients, [1, 2, 3])


def test_LincombDuneGridFunction_to_dune_grid_function():
    func = _make_dune_func(0.5, 'LincombDuneFunction', grid=grid)
    dune_func = to_dune_grid_function(func, grid, ensure_lincomb=True)
    assert dune_func == func
    dune_func = to_dune_grid_function(func, grid, ensure_lincomb=False)
    assert dune_func == func

def test_DuneGridFunction_to_dune_grid_function():
    func = _make_dune_func(0.5, 'DuneFunction', grid=grid)
    dune_func = to_dune_grid_function(func, grid, ensure_lincomb=False)
    assert dune_func == func
    dune_func = to_dune_grid_function(func, grid, ensure_lincomb=True)
    assert isinstance(dune_func, LincombDuneGridFunction)
    assert dune_func.functions[0] == func.impl

def test_LincombDuneFunction_to_dune_grid_function():
    func = _make_dune_func(0.5, 'LincombDuneFunction')
    for dune_func in (
            to_dune_grid_function(func, grid, ensure_lincomb=True),
            to_dune_grid_function(func, grid, ensure_lincomb=False)):
        assert isinstance(dune_func, LincombDuneGridFunction)
        assert isinstance(dune_func.functions[0], GridFunction2dCubeYaspgridTo1d)

def test_DuneFunction_to_dune_grid_function():
    func = _make_dune_func(0.5, 'DuneFunction')
    dune_func = to_dune_grid_function(func, grid, ensure_lincomb=False)
    assert isinstance(dune_func, DuneGridFunction)
    assert isinstance(dune_func.impl, GridFunction2dCubeYaspgridTo1d)
    dune_func = to_dune_grid_function(func, grid, ensure_lincomb=True)
    assert isinstance(dune_func, LincombDuneGridFunction)
    assert isinstance(dune_func.functions[0], GridFunction2dCubeYaspgridTo1d)

def test_convertible_function_to_dune_grid_function():
    f = ConstantFunction(value=1, dim_domain=2)
    for func in (f, f*f*f):
        func = to_dune_grid_function(f, grid)
        assert isinstance(func, DuneGridFunction)
        lincomb_func = to_dune_grid_function(f, grid, ensure_lincomb=True)
        assert isinstance(lincomb_func, LincombDuneGridFunction)
        for f_ in func.impl, lincomb_func.functions[0]:
            assert isinstance(f_, GridFunction2dCubeYaspgridTo1d)

def test_LincombFunction_to_dune_grid_function():
    f = ConstantFunction(value=1, dim_domain=2)
    pymor_func = f + 2*f + 3*f

    lincomb_func1 = to_dune_grid_function(pymor_func, grid)
    lincomb_func2 = to_dune_grid_function(pymor_func, grid, ensure_lincomb=True)

    for f in lincomb_func1, lincomb_func2:
        assert isinstance(f, LincombDuneGridFunction)
        assert all(isinstance(func, GridFunction2dCubeYaspgridTo1d) for func in f.functions)
        assert np.allclose(f.coefficients, [1, 2, 3])

def test_ExpressionFunction_to_dune_grid_function():
    f = ExpressionFunction('x[..., 0]', dim_domain=2)
    df = to_dune_grid_function(f, grid)
    assert isinstance(df, DuneGridFunction)

    assert np.allclose(f.evaluate(grid.centers(grid.dimension)), np.array(df.impl.dofs.vector, copy=False))

def test_parametric_ExpressionFunction_to_dune_grid_function():
    f = ExpressionFunction('foo[0]*x[..., 0]', dim_domain=2, parameters={'foo': 1})
    mu = f.parameters.parse(2)
    df = to_dune_grid_function(f, grid, mu=mu)
    assert isinstance(df, DuneGridFunction)

    assert np.allclose(f.evaluate(grid.centers(grid.dimension), mu=mu), np.array(df.impl.dofs.vector, copy=False))

def test_vector_valued_ExpressionFunction_to_dune_grid_function():
    f = ExpressionFunction('np.array([x[..., 0], x[..., 1]]).T', dim_domain=2, shape_range=(2,))
    df = to_dune_grid_function(f, grid)
    assert isinstance(df, DuneGridFunction)

    f_vals = f.evaluate(grid.centers(grid.dimension))
    f_vals = np.hstack([f_vals[:, 0], f_vals[:, 1]])

    assert np.allclose(f_vals, np.array(df.impl.dofs.vector, copy=False))
