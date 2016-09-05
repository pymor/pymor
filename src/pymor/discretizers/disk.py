# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import os.path
import configparser

import numpy as np

from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.discretizations.basic import StationaryDiscretization
from pymor.discretizations.basic import InstationaryDiscretization
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.spaces import CubicParameterSpace
from pymor.parameters.functionals import ExpressionParameterFunctional


def discretize_stationary_from_disk(parameter_file):
    """Load a linear affinely decomposed |StationaryDiscretization| from file.

    The discretization is defined via an `.ini`-style file as follows ::

        [system-matrices]
        L_1.mat: l_1(μ_1,...,μ_n)
        L_2.mat: l_2(μ_1,...,μ_n)
        ...

        [rhs-vectors]
        F_1.mat: f_1(μ_1,...,μ_n)
        F_2.mat: f_2(μ_1,...,μ_n)
        ...

        [parameter]
        μ_1: a_1,b_1
        ...
        μ_n: a_n,b_n

        [products]
        Prod1: P_1.mat
        Prod2: P_2.mat
        ...

    Here, `L_1.mat`, `L_2.mat`, ..., `F_1.mat`, `F_2.mat`, ... are files
    containing matrices `L_1`, `L_2`, ... and vectors `F_1.mat`, `F_2.mat`, ...
    which correspond to the affine components of the operator and right-hand
    side functional.  The respective coefficient functionals, are given via the
    string expressions `l_1(...)`, `l_2(...)`, ..., `f_1(...)` in the
    (scalar-valued) |Parameter| components `w_1`, ..., `w_n`. The allowed lower
    and upper bounds `a_i, b_i` for the component `μ_i` are specified in the
    `[parameters]` section. The resulting operator and right-hand side are
    then of the form ::

        L(μ) = l_1(μ)*L_1 + l_2(μ)*L_2+ ...
        F(μ) = f_1(μ)*F_1 + f_2(μ)*L_2+ ...

    In the `[products]` section, an optional list of inner products `Prod1`, `Prod2`, ..
    with corresponding matrices `P_1.mat`, `P_2.mat` can be specified.

    Example::

        [system-matrices]
        matrix1.mat: 1.
        matrix2.mat: 1. - theta**2

        [rhs-vectors]
        rhs.mat: 1.

        [parameter]
        theta: 0, 0.5

        [products]
        h1: h1.mat
        l2: mass.mat


    Parameters
    ----------
    parameter_file
        Path to the parameter file.

    Returns
    -------
    discretization
        The |StationaryDiscretization| that has been generated.
    """
    assert ".ini" == parameter_file[-4:], "Given file is not an .ini file"
    base_path = os.path.dirname(parameter_file)

    # Get input from parameter file
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(parameter_file)

    # Assert that all needed entries given
    assert 'system-matrices' in config.sections()
    assert 'rhs-vectors' in config.sections()
    assert 'parameter' in config.sections()

    system_mat = config.items('system-matrices')
    rhs_vec = config.items('rhs-vectors')
    parameter = config.items('parameter')

    # Dict of parameters types and ranges
    parameter_type = {}
    parameter_range = {}

    # get parameters
    for i in range(len(parameter)):
        parameter_name = parameter[i][0]
        parameter_list = tuple(float(j) for j in parameter[i][1].replace(" ", "").split(','))
        parameter_range[parameter_name] = parameter_list
        # Assume scalar parameter dependence
        parameter_type[parameter_name] = 0

    # Create parameter space
    parameter_space = CubicParameterSpace(parameter_type=parameter_type, ranges=parameter_range)

    # Assemble operators
    system_operators, system_functionals = [], []

    # get parameter functionals and system matrices
    for i in range(len(system_mat)):
        path = os.path.join(base_path, system_mat[i][0])
        expr = system_mat[i][1]
        parameter_functional = ExpressionParameterFunctional(expr, parameter_type=parameter_type)
        system_operators.append(NumpyMatrixOperator.from_file(path))
        system_functionals.append(parameter_functional)

    system_lincombOperator = LincombOperator(system_operators, coefficients=system_functionals)

    # get rhs vectors
    rhs_operators, rhs_functionals = [], []

    for i in range(len(rhs_vec)):
        path = os.path.join(base_path, rhs_vec[i][0])
        expr = rhs_vec[i][1]
        parameter_functional = ExpressionParameterFunctional(expr, parameter_type=parameter_type)
        op = NumpyMatrixOperator.from_file(path)
        assert isinstance(op._matrix, np.ndarray)
        op = op.with_(matrix=op._matrix.reshape((1, -1)))
        rhs_operators.append(op)
        rhs_functionals.append(parameter_functional)

    rhs_lincombOperator = LincombOperator(rhs_operators, coefficients=rhs_functionals)

    # get products if given
    if 'products' in config.sections():
        product = config.items('products')
        products = {}
        for i in range(len(product)):
            product_name = product[i][0]
            product_path = os.path.join(base_path, product[i][1])
            products[product_name] = NumpyMatrixOperator.from_file(product_path)
    else:
        products = None

    # Create and return stationary discretization
    return StationaryDiscretization(operator=system_lincombOperator, rhs=rhs_lincombOperator,
                                    parameter_space=parameter_space, products=products)


def discretize_instationary_from_disk(parameter_file, T=None, steps=None, u0=None, time_stepper=None):
    """Load a linear affinely decomposed |InstationaryDiscretization| from file.

    Similarly to :func:`discretize_stationary_from_disk`, the discretization is
    specified via an `ini.`-file of the following form ::

        [system-matrices]
        L_1.mat: l_1(μ_1,...,μ_n)
        L_2.mat: l_2(μ_1,...,μ_n)
        ...

        [rhs-vectors]
        F_1.mat: f_1(μ_1,...,μ_n)
        F_2.mat: f_2(μ_1,...,μ_n)
        ...

        [mass-matrix]
        D.mat

        [initial-solution]
        u0: u0.mat

        [parameter]
        μ_1: a_1,b_1
        ...
        μ_n: a_n,b_n

        [products]
        Prod1: P_1.mat
        Prod2: P_2.mat
        ...

        [time]
        T: final time
        steps: number of time steps


    Parameters
    ----------
    parameter_file
        Path to the '.ini' parameter file.
    T
        End-time of desired solution. If `None`, the value specified in the
        parameter file is used.
    steps
        Number of time steps to. If `None`, the value specified in the
        parameter file is used.
    u0
        Initial solution. If `None` the initial solution is obtained
        from parameter file.
    time_stepper
        The desired :class:`time stepper <pymor.algorithms.timestepping.TimeStepperInterface>`
        to use. If `None`, implicit Euler time stepping is used.

    Returns
    -------
    discretization
        The |InstationaryDiscretization| that has been generated.
    """
    assert ".ini" == parameter_file[-4:], "Given file is not an .ini file"
    base_path = os.path.dirname(parameter_file)

    # Get input from parameter file
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(parameter_file)

    # Assert that all needed entries given
    assert 'system-matrices' in config.sections()
    assert 'mass-matrix' in config.sections()
    assert 'rhs-vectors' in config.sections()
    assert 'parameter' in config.sections()

    system_mat = config.items('system-matrices')
    mass_mat = config.items('mass-matrix')
    rhs_vec = config.items('rhs-vectors')
    parameter = config.items('parameter')

    # Dict of parameters types and ranges
    parameter_type = {}
    parameter_range = {}

    # get parameters
    for i in range(len(parameter)):
        parameter_name = parameter[i][0]
        parameter_list = tuple(float(j) for j in parameter[i][1].replace(" ", "").split(','))
        parameter_range[parameter_name] = parameter_list
        # Assume scalar parameter dependence
        parameter_type[parameter_name] = 0

    # Create parameter space
    parameter_space = CubicParameterSpace(parameter_type=parameter_type, ranges=parameter_range)

    # Assemble operators
    system_operators, system_functionals = [], []

    # get parameter functionals and system matrices
    for i in range(len(system_mat)):
        path = os.path.join(base_path, system_mat[i][0])
        expr = system_mat[i][1]
        parameter_functional = ExpressionParameterFunctional(expr, parameter_type=parameter_type)
        system_operators.append(NumpyMatrixOperator.from_file(path))
        system_functionals.append(parameter_functional)

    system_lincombOperator = LincombOperator(system_operators, coefficients=system_functionals)

    # get rhs vectors
    rhs_operators, rhs_functionals = [], []

    for i in range(len(rhs_vec)):
        path = os.path.join(base_path, rhs_vec[i][0])
        expr = rhs_vec[i][1]
        parameter_functional = ExpressionParameterFunctional(expr, parameter_type=parameter_type)
        op = NumpyMatrixOperator.from_file(path)
        assert isinstance(op._matrix, np.ndarray)
        op = op.with_(matrix=op._matrix.reshape((1, -1)))
        rhs_operators.append(op)
        rhs_functionals.append(parameter_functional)

    rhs_lincombOperator = LincombOperator(rhs_operators, coefficients=rhs_functionals)

    # get mass matrix
    path = os.path.join(base_path, mass_mat[0][1])
    mass_operator = NumpyMatrixOperator.from_file(path)

    # Obtain initial solution if not given
    if u0 is None:
        u_0 = config.items('initial-solution')
        path = os.path.join(base_path, u_0[0][1])
        op = NumpyMatrixOperator.from_file(path)
        assert isinstance(op._matrix, np.ndarray)
        u0 = op.with_(matrix=op._matrix.reshape((-1, 1)))

    # get products if given
    if 'products' in config.sections():
        product = config.items('products')
        products = {}
        for i in range(len(product)):
            product_name = product[i][0]
            product_path = os.path.join(base_path, product[i][1])
            products[product_name] = NumpyMatrixOperator.from_file(product_path)
    else:
        products = None

    # Further specifications
    if 'time' in config.sections():
        if T is None:
            assert 'T' in config.options('time')
            T = float(config.get('time', 'T'))
        if steps is None:
            assert 'steps' in config.options('time')
            steps = int(config.get('time', 'steps'))

    # Use implicit euler time stepper if no time-stepper given
    if time_stepper is None:
        time_stepper = ImplicitEulerTimeStepper(steps)
    else:
        time_stepper = time_stepper(steps)

    # Create and return instationary discretization
    return InstationaryDiscretization(operator=system_lincombOperator, rhs=rhs_lincombOperator,
                                      parameter_space=parameter_space, initial_data=u0, T=T,
                                      time_stepper=time_stepper, mass=mass_operator, products=products)
