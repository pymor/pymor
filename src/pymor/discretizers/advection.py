# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Falk Meyer <falk.meyer@wwu.de>

from __future__ import absolute_import, division, print_function
from scipy import io

import numpy as np
import ConfigParser

from pymor.algorithms.timestepping import ExplicitEulerTimeStepper
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.discretizations.basic import InstationaryDiscretization
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.gui.qt import PatchVisualizer, Matplotlib1DVisualizer
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyGenericOperator
from pymor.operators.numpy import NumpyMatrixOperator 
from pymor.operators.fv import (nonlinear_advection_lax_friedrichs_operator,
                                nonlinear_advection_engquist_osher_operator,
                                nonlinear_advection_simplified_engquist_osher_operator,
                                L2Product, L2ProductFunctional)
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.parameters.spaces import CubicParameterSpace
from pymor.parameters.functionals import ExpressionParameterFunctional

def discretize_nonlinear_instationary_advection_fv(analytical_problem, diameter=None, nt=100, num_flux='lax_friedrichs',
                                                   lxf_lambda=1., eo_gausspoints=5, eo_intervals=1, num_values=None,
                                                   domain_discretizer=None, grid=None, boundary_info=None):
    """Discretizes an |InstationaryAdvectionProblem| using the finite volume method.

    Simple explicit Euler time-stepping is used for time-discretization.

    Parameters
    ----------
    analytical_problem
        The |InstationaryAdvectionProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed to the `domain_discretizer`.
    nt
        The number of time-steps.
    num_flux
        The numerical flux to use in the finite volume formulation. Allowed
        values are `'lax_friedrichs'`, `'engquist_osher'`, `'simplified_engquist_osher'`.
        (See :mod:`pymor.operators.fv`.)
    lxf_lambda
        The stabilization parameter for the Lax-Friedrichs numerical flux.
        (Ignored, if different flux is chosen.)
    eo_gausspoints
        Number of Gauss points for the Engquist-Osher numerical flux.
        (Ignored, if different flux is chosen.)
    eo_intervals
        Number of sub-intervals to use for integration when using Engquist-Osher
        numerical flux. (Ignored, if different flux is chosen.)
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If further arguments should be passed to the discretizer, use
        :func:`functools.partial`. If `None`, |discretize_domain_default| is used.
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is specified.

    Returns
    -------
    discretization
        The |Discretization| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.


    """

    assert isinstance(analytical_problem, InstationaryAdvectionProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None
    assert num_flux in ('lax_friedrichs', 'engquist_osher', 'simplified_engquist_osher')

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    p = analytical_problem

    if num_flux == 'lax_friedrichs':
        L = nonlinear_advection_lax_friedrichs_operator(grid, boundary_info, p.flux_function,
                                                        dirichlet_data=p.dirichlet_data, lxf_lambda=lxf_lambda)
    elif num_flux == 'engquist_osher':
        L = nonlinear_advection_engquist_osher_operator(grid, boundary_info, p.flux_function,
                                                        p.flux_function_derivative,
                                                        gausspoints=eo_gausspoints, intervals=eo_intervals,
                                                        dirichlet_data=p.dirichlet_data)
    else:
        L = nonlinear_advection_simplified_engquist_osher_operator(grid, boundary_info, p.flux_function,
                                                                   p.flux_function_derivative,
                                                                   dirichlet_data=p.dirichlet_data)
    F = None if p.rhs is None else L2ProductFunctional(grid, p.rhs)

    if p.initial_data.parametric:
        def initial_projection(U, mu):
            I = p.initial_data.evaluate(grid.quadrature_points(0, order=2), mu).squeeze()
            I = np.sum(I * grid.reference_element.quadrature(order=2)[1], axis=1) * (1. / grid.reference_element.volume)
            I = NumpyVectorArray(I, copy=False)
            return I.lincomb(U).data
        I = NumpyGenericOperator(initial_projection, dim_range=grid.size(0), linear=True,
                                 parameter_type=p.initial_data.parameter_type)
    else:
        I = p.initial_data.evaluate(grid.quadrature_points(0, order=2)).squeeze()
        I = np.sum(I * grid.reference_element.quadrature(order=2)[1], axis=1) * (1. / grid.reference_element.volume)
        I = NumpyVectorArray(I, copy=False)

    products = {'l2': L2Product(grid, boundary_info)}
    if grid.dim == 2:
        visualizer = PatchVisualizer(grid=grid, bounding_box=grid.domain, codim=0)
    elif grid.dim == 1:
        visualizer = Matplotlib1DVisualizer(grid, codim=0)
    else:
        visualizer = None
    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None
    time_stepper = ExplicitEulerTimeStepper(nt=nt)

    discretization = InstationaryDiscretization(operator=L, rhs=F, initial_data=I, T=p.T, products=products,
                                                time_stepper=time_stepper,
                                                parameter_space=parameter_space, visualizer=visualizer,
                                                num_values=num_values, name='{}_FV'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}

def discretize_nonlinear_instationary_advection_disk(parameterFile, T = None, steps = None, u0 = None, time_stepper = None):
    """Generates instationary discretization only based on system relevant data given as .mat files on the hard disc. The path and further
    specifications to these objects are given in an '.ini' parameter file (see example below). Suitable for discrete problems given by::

    M * /delta_t u(t, w) + L(u(w), t, w) = F(t, w)
                            u(0, w) = u_0(w)

    for t in [0,T], where L is a (possibly non-linear) time-dependent
    |Operator|, F is a time-dependent linear |Functional|, u_0 the
    initial data and w the parameter. The mass |Operator| M is assumed to be linear,
    time-independent and |Parameter|-independent.

    Parameters
    ----------
    parameterFile
	    String containing the path to the '.ini' parameter file.
	T
        Endtime of desired solution, if None obtained from parameter file
    steps
        Number of time steps to do, if None obtained from parameter file
    u0
	    Initial solution, if None obtained from parameter file
	time_stepper
	    The desired time_stepper to use, , if None an Implicit euler scheme is used.

    Returns
    -------
    discretization
        The |Discretization| that has been generated.

    Example parameter_file
    -------
    Following parameter file is suitable for a discrete parabolic problem with

    (u(w), w) = (f_1(w)*K1 + f_2(w)*K2+...)*u, F(w) = g_1(w)*L1+g_2(w)*L2+..., M = D and
    u_0(w)=u0 with parameter w_i in [a_i,b_i], where f_i(w) and g_i(w) are strings of valid python
    expressions.

    Optional products can be provided to introduce a dict of inner products on the discrete space.
    Time specifications like T and steps can also be provided, but are optional when already given
    by call of this method. The content of the file is then given as:

    [stiffness-matrices]
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
    assert ".ini" == parameterFile[-4:], "Given file is not an .ini file"

    # Get input from parameter file
    config = ConfigParser.ConfigParser()
    config.optionxform = str
    config.read(parameterFile)

    # Assert that all needed entries given
    assert 'stiffness-matrices' in config.sections()
    assert 'mass-matrix' in config.sections()
    assert 'rhs-vectors' in config.sections()
    assert 'parameter' in config.sections()

    stiff_mat        = config.items('stiffness-matrices')
    mass_mat         = config.items('mass-matrix')
    rhs_vec          = config.items('rhs-vectors')
    parameter        = config.items('parameter')

    # Dict of parameters types and ranges
    parameter_type = {}
    parameter_range = {}

    # get parameters
    for i in range(len(parameter)):
        parameter_name = parameter[i][0]
        parameter_list = tuple(float(j) for j in parameter[i][1].replace(" ","").split(','))
        parameter_range[parameter_name] = parameter_list
        # Assume scalar parameter dependence
        parameter_type[parameter_name] = 0

    # Create parameter space
    parameter_space = CubicParameterSpace(parameter_type=parameter_type,ranges=parameter_range)

    # Assemble operators
    stiff_operators, stiff_functionals = [], []

    # get parameter functionals and stiffness matrices
    for i in range(len(stiff_mat)):
        path = stiff_mat[i][0]
        expr = stiff_mat[i][1]
        parameter_functional = ExpressionParameterFunctional(expr, parameter_type=parameter_type)
        info = io.loadmat(path, mat_dtype=True).values()
        stiff_operators.append(NumpyMatrixOperator(info[[j for j in range(len(info)) if type(info[j]) == np.ndarray][0]]))
        stiff_functionals.append(parameter_functional)

    stiff_lincombOperator = LincombOperator(stiff_operators, coefficients=stiff_functionals)

    # get rhs vectors
    rhs_operators, rhs_functionals = [], []

    for i in range(len(rhs_vec)):
        path = rhs_vec[i][0]
        expr = rhs_vec[i][1]
        parameter_functional = ExpressionParameterFunctional(expr, parameter_type=parameter_type)
        info = io.loadmat(path, mat_dtype=True).values()
        rhs_operators.append(NumpyMatrixOperator(info[[j for j in range(len(info)) if type(info[j]) == np.ndarray][0]].T))
        rhs_functionals.append(parameter_functional)

    rhs_lincombOperator = LincombOperator(rhs_operators, coefficients=rhs_functionals)

    # get mass matrix
    path = mass_mat[0][1]
    info = io.loadmat(path, mat_dtype=True).values()
    mass_operator = NumpyMatrixOperator(info[[j for j in range(len(info)) if type(info[j]) == np.ndarray][0]])

    # Obtain initial solution if not given
    if u0 is None:
        u_0 = config.items('initial-solution')
        path = u_0[0][1]
        info = io.loadmat(path, mat_dtype=True).values()
        u0  = NumpyMatrixOperator(info[[j for j in range(len(info)) if type(info[j]) == np.ndarray][0]])

    # get products if given
    if 'products' in config.sections():
        product = config.items('products')
        products = {}
        for i in range(len(product)):
            product_name = product[i][0]
            product_path = product[i][1]
            info=io.loadmat(product_path,mat_dtype=True).values()
            products[product_name] = NumpyMatrixOperator(info[[j for j in range(len(info)) if type(info[j]) == np.ndarray][0]])
    else:
        products = None

    # Further specifications
    if 'time' in config.sections():
        if T is None:
            assert 'T' in config.options('time')
            T = float(config.get('time','T'))
        if steps is None:
            assert 'steps' in config.options('time')
            steps = int(config.get('time','steps'))

    # Use implicit euler time stepper if no time-stepper given
    if time_stepper is None:
        time_stepper = ImplicitEulerTimeStepper(steps)
    else:
        time_stepper = time_stepper(steps)

    # Create and return instationary discretization
    return InstationaryDiscretization(operator=stiff_lincombOperator, rhs=rhs_lincombOperator, parameter_space=parameter_space, initial_data=u0, T=T, time_stepper=time_stepper, mass=mass_operator, products=products)
 
