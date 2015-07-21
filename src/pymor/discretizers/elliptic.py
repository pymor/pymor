# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: lucas-ca <lucascamp@web.de>
#		Falk Meyer <falk.meyer@wwu.de>

from __future__ import absolute_import, division, print_function
from scipy import io

import numpy as np
import ConfigParser

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.discretizations.basic import StationaryDiscretization
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.grids.oned import OnedGrid
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid
from pymor.gui.qt import PatchVisualizer, Matplotlib1DVisualizer
from pymor.operators import cg, fv
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace

def discretize_elliptic_cg(analytical_problem, diameter=None, domain_discretizer=None,
                           grid=None, boundary_info=None):
    """Discretizes an |EllipticProblem| using finite elements.

    Parameters
    ----------
    analytical_problem
        The |EllipticProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed to the `domain_discretizer`.
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

    assert isinstance(analytical_problem, EllipticProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    assert isinstance(grid, (OnedGrid, TriaGrid, RectGrid))

    if isinstance(grid, RectGrid):
        Operator = cg.DiffusionOperatorQ1
        Functional = cg.L2ProductFunctionalQ1
    else:
        Operator = cg.DiffusionOperatorP1
        Functional = cg.L2ProductFunctionalP1

    p = analytical_problem

    if p.diffusion_functionals is not None:
        L0 = Operator(grid, boundary_info, diffusion_constant=0, name='diffusion_boundary_part')

        Li = [Operator(grid, boundary_info, diffusion_function=df, dirichlet_clear_diag=True,
                       name='diffusion_{}'.format(i))
              for i, df in enumerate(p.diffusion_functions)]

        L = LincombOperator(operators=[L0] + Li, coefficients=[1.] + list(p.diffusion_functionals),
                            name='diffusion')
    else:
        assert len(p.diffusion_functions) == 1
        L = Operator(grid, boundary_info, diffusion_function=p.diffusion_functions[0],
                     name='diffusion')

    F = Functional(grid, p.rhs, boundary_info, dirichlet_data=p.dirichlet_data, neumann_data=p.neumann_data)

    if isinstance(grid, (TriaGrid, RectGrid)):
        visualizer = PatchVisualizer(grid=grid, bounding_box=grid.domain, codim=2)
    else:
        visualizer = Matplotlib1DVisualizer(grid=grid, codim=1)

    empty_bi = EmptyBoundaryInfo(grid)
    l2_product = cg.L2ProductQ1(grid, empty_bi) if isinstance(grid, RectGrid) else cg.L2ProductP1(grid, empty_bi)
    h1_semi_product = Operator(grid, empty_bi)
    products = {'h1': l2_product + h1_semi_product,
                'h1_semi': h1_semi_product,
                'l2': l2_product}

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization = StationaryDiscretization(L, F, products=products, visualizer=visualizer,
                                              parameter_space=parameter_space, name='{}_CG'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}


def discretize_elliptic_fv(analytical_problem, diameter=None, domain_discretizer=None,
                           grid=None, boundary_info=None):
    """Discretizes an |EllipticProblem| using the finite volume method.

    Parameters
    ----------
    analytical_problem
        The |EllipticProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed to the `domain_discretizer`.
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

    assert isinstance(analytical_problem, EllipticProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    p = analytical_problem

    if p.diffusion_functionals is not None:
        Li = [fv.DiffusionOperator(grid, boundary_info, diffusion_function=df, name='diffusion_{}'.format(i))
              for i, df in enumerate(p.diffusion_functions)]
        L = LincombOperator(operators=Li, coefficients=list(p.diffusion_functionals),
                            name='diffusion')

        F0 = fv.L2ProductFunctional(grid, p.rhs, boundary_info=boundary_info, neumann_data=p.neumann_data)
        if p.dirichlet_data is not None:
            Fi = [fv.L2ProductFunctional(grid, None, boundary_info=boundary_info, dirichlet_data=p.dirichlet_data,
                                         diffusion_function=df, name='dirichlet_{}'.format(i))
                  for i, df in enumerate(p.diffusion_functions)]
            F = LincombOperator(operators=[F0] + Fi, coefficients=[1.] + list(p.diffusion_functionals),
                                name='rhs')
        else:
            F = F0

    else:
        assert len(p.diffusion_functions) == 1
        L = fv.DiffusionOperator(grid, boundary_info, diffusion_function=p.diffusion_functions[0],
                                 name='diffusion')

        F = fv.L2ProductFunctional(grid, p.rhs, boundary_info=boundary_info, dirichlet_data=p.dirichlet_data,
                                   diffusion_function=p.diffusion_functions[0], neumann_data=p.neumann_data)

    if isinstance(grid, (TriaGrid, RectGrid)):
        visualizer = PatchVisualizer(grid=grid, bounding_box=grid.domain, codim=0)
    elif isinstance(grid, (OnedGrid)):
        visualizer = Matplotlib1DVisualizer(grid=grid, codim=0)
    else:
        visualizer = None

    l2_product = fv.L2Product(grid)
    products = {'l2': l2_product}

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization = StationaryDiscretization(L, F, products=products, visualizer=visualizer,
                                              parameter_space=parameter_space, name='{}_FV'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}

def discretize_elliptic_disk(parameter_file):
    """Generates stationary discretization only based on system relevant data given as .mat files on the hard disc. The path and further
     specifications to these objects are given in an '.ini' parameter file (see example below). Suitable for discrete problems given by::

	 L(u(w), w) = F(w)

    with an operator L and a linear functional F with a parameter w  given as stiffness matrices and rhs vectors in an affine decomposition 
    on the hard disc.

    Parameters
    ----------
    parameterFile
	    String containing the path to the .ini parameterfile.

    Returns
    -------
    discretization
        The |Discretization| that has been generated.


    Example parameter_file
    -------
    Following parameter file is suitable for a discrete elliptic problem with

    L(u(w), w) = (f_1(w)*K1 + f_2(w)*K2+...)*u and F(w) = g_1(w)*L1+g_2(w)*L2+... with
    parameter w_i in [a_i,b_i], where f_i(w) and g_i(w) are strings of valid python
    expressions.

    Optional products can be provided to introduce a dict of inner products on
    the discrete space. The content of the file is then given as:

    [stiffness-matrices]
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
    assert 'stiffness-matrices' in config.sections()
    assert 'rhs-vectors' in config.sections()
    assert 'parameter' in config.sections()

    stiff_mat        = config.items('stiffness-matrices')
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

    # Create and return stationary discretization
    return StationaryDiscretization(operator=stiff_lincombOperator, rhs=rhs_lincombOperator, parameter_space=parameter_space, products=products)
