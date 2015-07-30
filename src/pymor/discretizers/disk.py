# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Falk Meyer <falk.meyer@wwu.de>

from __future__ import absolute_import, division, print_function
from scipy import io

import numpy as np
import ConfigParser

from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.discretizations.basic import StationaryDiscretization
from pymor.discretizations.basic import InstationaryDiscretization
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.spaces import CubicParameterSpace
from pymor.parameters.functionals import ExpressionParameterFunctional


def discretize_stationary_from_disk(parameter_file):
    """Generates stationary discretization only based on system relevant data given as .mat files on the hard disc.
    The path and further specifications to these objects are given in an '.ini' parameter file (see example below).
    Suitable for discrete problems given by::

    L(u, w) = F(w)

    with an operator L and a linear functional F with a parameter w  given as system matrices and rhs vectors in
    an affine decomposition on the hard disk.

    Parameters
    ----------
    parameterFile
        String containing the path to the .ini parameter file.

    Returns
    -------
    discretization
        The |Discretization| that has been generated.


    Example parameter_file
    -------
    Following parameter file is suitable for a discrete elliptic problem with

    L(u, w) = (f_1(w)*K1 + f_2(w)*K2+...)*u and F(w) = g_1(w)*L1+g_2(w)*L2+... with
    parameter w_i in [a_i,b_i], where f_i(w) and g_i(w) are strings of valid python
    expressions.

    Optional products can be provided to introduce a dict of inner products on
    the discrete space. The content of the file is then given as:

    [system-matrices]
    # path_to_object: parameter_functional_associated_with_object
    K1.mat: f_1(w_1,...,w_n)
    K2.mat: f_2(w_1,...,w_n)
    ...
    [rhs-vectors]
    L1.mat: g_1(w_1,...,w_n)
    L2.mat: g_2(w_1,...,w_n)
    ...
    [parameter]
    # Name: lower_bound,upper_bound
    w_1: a_1,b_1
    ...
    w_n: a_n,b_n
    [products]
    # Name: path_to_object
    Prod1: S.mat
    Prod2: T.mat
    ...
    """
    assert ".ini" == parameter_file[-4:], "Given file is not an .ini file"

    # Get input from parameter file
    config = ConfigParser.ConfigParser()
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
        path = system_mat[i][0]
        expr = system_mat[i][1]
        parameter_functional = ExpressionParameterFunctional(expr, parameter_type=parameter_type)
        info = io.loadmat(path, mat_dtype=True).values()
        system_operators.append(NumpyMatrixOperator([j for j in info if isinstance(j, np.ndarray)][0]))
        system_functionals.append(parameter_functional)

    system_lincombOperator = LincombOperator(system_operators, coefficients=system_functionals)

    # get rhs vectors
    rhs_operators, rhs_functionals = [], []

    for i in range(len(rhs_vec)):
        path = rhs_vec[i][0]
        expr = rhs_vec[i][1]
        parameter_functional = ExpressionParameterFunctional(expr, parameter_type=parameter_type)
        info = io.loadmat(path, mat_dtype=True).values()
        rhs_operators.append(NumpyMatrixOperator([j for j in info if isinstance(j, np.ndarray)][0].T))
        rhs_functionals.append(parameter_functional)

    rhs_lincombOperator = LincombOperator(rhs_operators, coefficients=rhs_functionals)

    # get products if given
    if 'products' in config.sections():
        product = config.items('products')
        products = {}
        for i in range(len(product)):
            product_name = product[i][0]
            product_path = product[i][1]
            info = io.loadmat(product_path, mat_dtype=True).values()
            products[product_name] = NumpyMatrixOperator([j for j in info if isinstance(j, np.ndarray)][0])
    else:
        products = None

    # Create and return stationary discretization
    return StationaryDiscretization(operator=system_lincombOperator, rhs=rhs_lincombOperator,
                                    parameter_space=parameter_space, products=products)


def discretize_instationary_from_disk(parameter_file, T=None, steps=None, u0=None, time_stepper=None):
    """Generates instationary discretization only based on system relevant data given as .mat files
    on the hard disc. The path and further specifications to these objects are given in an '.ini'
    parameter file (see example below). Suitable for discrete problems given by::

    M(u(t), w) + L(u(t), w, t) = F(t, w)
                          u(0) = u_0

    for t in [0,T], where L is a linear time-dependent
    |Operator|, F is a time-dependent linear |Functional|, u_0 the
    initial data and w the parameter. The mass |Operator| M is assumed to be linear,
    time-independent and |Parameter|-independent.

    Parameters
    ----------
    parameter_file
        String containing the path to the '.ini' parameter file.
    T
        End-time of desired solution, if None obtained from parameter file
    steps
        Number of time steps to do, if None obtained from parameter file
    u0
        Initial solution, if None obtained from parameter file
    time_stepper
        The desired time_stepper to use, if None an Implicit euler scheme is used.

    Returns
    -------
    discretization
        The |Discretization| that has been generated.

    Example parameter_file
    -------
    Following parameter file is suitable for a discrete parabolic problem with

    L(u(w), w) = (f_1(w)*K1 + f_2(w)*K2+...)*u, F(w) = g_1(w)*L1+g_2(w)*L2+..., M = D and
    u_0(w)=u0 with parameter w_i in [a_i,b_i], where f_i(w) and g_i(w) are strings of valid python
    expressions.

    Optional products can be provided to introduce a dict of inner products on the discrete space.
    Time specifications like T and steps can also be provided, but are optional when already given
    by call of this method. The content of the file is then given as:

    [system-matrices]
    # path_to_object: parameter_functional_associated_with_object
    K1.mat: f_1(w_1,...,w_n)
    K2.mat: f_2(w_1,...,w_n)
    ...
    [rhs-vectors]
    L1.mat: g_1(w_1,...,w_n)
    L2.mat: g_2(w_1,...,w_n)
    ...
    [mass-matrix]
    D.mat
    [initial-solution]
    u0: u0.mat
    [parameter]
    # Name: lower_bound,upper_bound
    w_1: a_1,b_1
    ...
    w_n: a_n,b_n
    [products]
    # Name: path_to_object
    Prod1: S.mat
    Prod2: T.mat
    ...
    [time]
    # fixed_Name: value
    T: 10.0
    steps: 100
    """
    assert ".ini" == parameter_file[-4:], "Given file is not an .ini file"

    # Get input from parameter file
    config = ConfigParser.ConfigParser()
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
        path = system_mat[i][0]
        expr = system_mat[i][1]
        parameter_functional = ExpressionParameterFunctional(expr, parameter_type=parameter_type)
        info = io.loadmat(path, mat_dtype=True).values()
        system_operators.append(NumpyMatrixOperator([j for j in info if isinstance(j, np.ndarray)][0]))
        system_functionals.append(parameter_functional)

    system_lincombOperator = LincombOperator(system_operators, coefficients=system_functionals)

    # get rhs vectors
    rhs_operators, rhs_functionals = [], []

    for i in range(len(rhs_vec)):
        path = rhs_vec[i][0]
        expr = rhs_vec[i][1]
        parameter_functional = ExpressionParameterFunctional(expr, parameter_type=parameter_type)
        info = io.loadmat(path, mat_dtype=True).values()
        rhs_operators.append(NumpyMatrixOperator([j for j in info if isinstance(j, np.ndarray)][0].T))
        rhs_functionals.append(parameter_functional)

    rhs_lincombOperator = LincombOperator(rhs_operators, coefficients=rhs_functionals)

    # get mass matrix
    path = mass_mat[0][1]
    info = io.loadmat(path, mat_dtype=True).values()
    mass_operator = NumpyMatrixOperator([j for j in info if isinstance(j, np.ndarray)][0])

    # Obtain initial solution if not given
    if u0 is None:
        u_0 = config.items('initial-solution')
        path = u_0[0][1]
        info = io.loadmat(path, mat_dtype=True).values()
        u0 = NumpyMatrixOperator([j for j in info if isinstance(j, np.ndarray)][0])

    # get products if given
    if 'products' in config.sections():
        product = config.items('products')
        products = {}
        for i in range(len(product)):
            product_name = product[i][0]
            product_path = product[i][1]
            info = io.loadmat(product_path, mat_dtype=True).values()
            products[product_name] = NumpyMatrixOperator([j for j in info if isinstance(j, np.ndarray)][0])
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
