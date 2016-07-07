# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module imports some commonly used methods and classes.

You can use ``from pymor.basic import *`` in interactive session
to have the most important parts of pyMOR directly available.
"""

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.basisextension import trivial_basis_extension, gram_schmidt_basis_extension, pod_basis_extension
from pymor.algorithms.ei import interpolate_operators, ei_greedy, deim
from pymor.algorithms.error import reduction_error_analysis
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.greedy import greedy
from pymor.algorithms.adaptivegreedy import adaptive_greedy
from pymor.algorithms.newton import newton
from pymor.algorithms.pod import pod

from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.analyticalproblems.burgers import BurgersProblem, Burgers2DProblem
from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.analyticalproblems.parabolic import ParabolicProblem
from pymor.analyticalproblems.thermalblock import ThermalBlockProblem

from pymor.core.cache import clear_caches, enable_caching, disable_caching
from pymor.core.defaults import print_defaults, write_defaults_to_file, load_defaults_from_file, set_defaults
from pymor.core.logger import set_log_levels, getLogger
from pymor.core.pickle import dump, dumps, load, loads

from pymor.discretizations.basic import StationaryDiscretization, InstationaryDiscretization

from pymor.domaindescriptions.basic import RectDomain, CylindricalDomain, TorusDomain, LineDomain, CircleDomain
from pymor.domaindescriptions.boundarytypes import BoundaryType
from pymor.domaindescriptions.polygonal import DiscDomain, CircularSectorDomain, PolygonalDomain

from pymor.domaindiscretizers.default import discretize_domain_default

from pymor.discretizers.advection import discretize_nonlinear_instationary_advection_fv
from pymor.discretizers.elliptic import discretize_elliptic_cg, discretize_elliptic_fv
from pymor.discretizers.parabolic import discretize_parabolic_cg, discretize_parabolic_fv

from pymor.functions.basic import ConstantFunction, GenericFunction, ExpressionFunction, LincombFunction
from pymor.functions.bitmap import BitmapFunction

from pymor.grids.boundaryinfos import EmptyBoundaryInfo, BoundaryInfoFromIndicators, AllDirichletBoundaryInfo
from pymor.grids.oned import OnedGrid
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid

from pymor.operators.constructions import (LincombOperator, Concatenation, ComponentProjection, IdentityOperator,
                                           ConstantOperator, ZeroOperator, VectorArrayOperator, VectorOperator,
                                           VectorFunctional, FixedParameterOperator, AdjointOperator,
                                           SelectionOperator, induced_norm)
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.operators.numpy import NumpyGenericOperator, NumpyMatrixOperator

from pymor.parallel.default import new_parallel_pool
from pymor.parallel.manager import RemoteObjectManager

from pymor.parameters.base import Parameter
from pymor.parameters.functionals import (ProjectionParameterFunctional, GenericParameterFunctional,
                                          ExpressionParameterFunctional)
from pymor.parameters.spaces import CubicParameterSpace

from pymor.reductors.basic import reduce_generic_rb, reduce_to_subbasis
from pymor.reductors.coercive import reduce_coercive, reduce_coercive_simple
from pymor.reductors.parabolic import reduce_parabolic

from pymor.tools.random import new_random_state

from pymor.vectorarrays.constructions import cat_arrays
from pymor.vectorarrays.interfaces import VectorSpace
from pymor.vectorarrays.list import ListVectorArray
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace
